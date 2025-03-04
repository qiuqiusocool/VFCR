import cupy as cp
from cupy.fft import fftn, ifftn
import time
from scipy.ndimage import zoom, map_coordinates
from skimage.registration import phase_cross_correlation
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# 计算地形阴影
def hillshade(array, azimuth, angle_altitude):
    x, y = np.gradient(array)
    slope = np.pi / 2. - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)
    azimuth_rad = azimuth * np.pi / 180.
    altitude_rad = angle_altitude * np.pi / 180.

    shaded = np.sin(altitude_rad) * np.sin(slope) + np.cos(altitude_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect)
    shaded = shaded.clip(0, 1)
    return shaded


# 绘制结果
def plot_results(original_DEM1, original_DEM2, reconstructed_DEM):
    fig1, axes1 = plt.subplots(2, 3, figsize=(24, 12))
    norm1 = Normalize(vmin=np.nanmin(original_DEM1), vmax=np.nanmax(original_DEM1))
    norm2 = Normalize(vmin=np.nanmin(original_DEM2), vmax=np.nanmax(original_DEM2))
    norm_rec = Normalize(vmin=np.nanmin(reconstructed_DEM), vmax=np.nanmax(reconstructed_DEM))
    azimuth = 325
    altitude = 45

    axes1[0, 0].imshow(original_DEM1, cmap='jet', norm=norm1)
    axes1[0, 0].set_title('Original DEM 1')
    axes1[0, 0].axis('off')

    axes1[0, 1].imshow(original_DEM2, cmap='jet', norm=norm2)
    axes1[0, 1].set_title('Original DEM 2')
    axes1[0, 1].axis('off')

    axes1[0, 2].imshow(reconstructed_DEM, cmap='jet', norm=norm_rec)
    axes1[0, 2].set_title('Reconstructed DEM')
    axes1[0, 2].axis('off')

    original_shaded1 = hillshade(original_DEM1, azimuth, altitude)
    axes1[1, 0].imshow(original_shaded1, cmap='gray')
    axes1[1, 0].set_title('Original DEM 1 Hillshade')
    axes1[1, 0].axis('off')

    original_shaded2 = hillshade(original_DEM2, azimuth, altitude)
    axes1[1, 1].imshow(original_shaded2, cmap='gray')
    axes1[1, 1].set_title('Original DEM 2 Hillshade')
    axes1[1, 1].axis('off')

    reconstructed_shaded = hillshade(reconstructed_DEM, azimuth, altitude)
    axes1[1, 2].imshow(reconstructed_shaded, cmap='gray')
    axes1[1, 2].set_title('Reconstructed DEM Hillshade')
    axes1[1, 2].axis('off')

    plt.tight_layout()
    plt.show()


# 定义掩码
def definemask(N):
    return np.ones((N, N))


# 水平方向上的前向差分
def dxf(x):
    return cp.roll(x, -1, axis=0) - x


# 垂直方向上的前向差分
def dyf(x):
    return cp.roll(x, -1, axis=1) - x


# 水平方向上的后向差分
def dxb(x):
    return x - cp.roll(x, 1, axis=0)


# 垂直方向上的后向差分
def dyb(x):
    return x - cp.roll(x, 1, axis=1)


# 配准
def register_images(ref_image, mov_image):
    ref_mask = ~np.isnan(ref_image)
    mov_mask = ~np.isnan(mov_image)

    temp_ref_image = np.copy(ref_image)
    temp_mov_image = np.copy(mov_image)
    temp_ref_image[~ref_mask] = 0
    temp_mov_image[~mov_mask] = 0

    shift_estimation = phase_cross_correlation(
        temp_ref_image, temp_mov_image,
        reference_mask=ref_mask,
        moving_mask=mov_mask,
        upsample_factor=10
    )[0]

    coords = np.meshgrid(np.arange(mov_image.shape[0]), np.arange(mov_image.shape[1]), indexing='ij')
    coords = np.array(coords) - np.reshape(shift_estimation, (1, 1, 1))

    shifted_image = map_coordinates(mov_image, coords, order=1, mode='constant', cval=np.nan)

    valid_mask = map_coordinates(mov_mask.astype(float), coords, order=1, mode='constant', cval=0)
    valid_mask = valid_mask > 0.5

    final_shifted_image = np.full_like(mov_image, np.nan)
    final_shifted_image[valid_mask] = mov_image[valid_mask]

    return final_shifted_image


def resample_to_low_res(x, scale_factor):
    return zoom(x, scale_factor, order=1)


def compute_curvature(dem):
    dzdx = np.gradient(dem, axis=1)
    dzdy = np.gradient(dem, axis=0)
    dzdxx = np.gradient(dzdx, axis=1)
    dzdyy = np.gradient(dzdy, axis=0)
    dzdxy = np.gradient(dzdx, axis=0)

    curvature = (dzdxx * (1 + dzdy ** 2) - 2 * dzdx * dzdy * dzdxy + dzdyy * (1 + dzdx ** 2)) / (
            (1 + dzdx ** 2 + dzdy ** 2) ** 1.5)
    return curvature


def detect_anomalies(dem, curvature_lower_threshold, curvature_upper_threshold):
    curvature = compute_curvature(dem)

    anomalies = np.zeros_like(dem, dtype=int)
    anomalies[(curvature > curvature_upper_threshold) | (curvature < curvature_lower_threshold)] = 1

    return anomalies


def delta_surface_fill(dem, anomalies, iterations=20):
    x, y = np.indices(dem.shape)
    valid_points = ~np.isnan(dem)
    points = np.c_[x[valid_points], y[valid_points]]
    values = dem[valid_points]

    tri = Delaunay(points)
    filled_dem = dem.copy()

    nan_points = np.c_[x[np.isnan(dem)], y[np.isnan(dem)]]
    initial_fill = griddata(points, values, nan_points, method='nearest')
    filled_dem[np.isnan(dem)] = initial_fill

    for _ in range(iterations):
        new_filled_dem = filled_dem.copy()
        for i in range(1, filled_dem.shape[0] - 1):
            for j in range(1, filled_dem.shape[1] - 1):
                if anomalies[i, j] == 1 or np.isnan(filled_dem[i, j]):
                    surrounding_values = [
                        filled_dem[i - 1, j],
                        filled_dem[i + 1, j],
                        filled_dem[i, j - 1],
                        filled_dem[i, j + 1],
                    ]
                    surrounding_values = [v for v in surrounding_values if not np.isnan(v)]
                    if surrounding_values:
                        new_filled_dem[i, j] = np.mean(surrounding_values)
        filled_dem = new_filled_dem
    return filled_dem


def fill_nans_with_delta_surface_fill(dem, anomalies):
    return delta_surface_fill(dem, anomalies)


def compute_curvature(dem):
    dzdx = np.gradient(dem, axis=1)
    dzdy = np.gradient(dem, axis=0)
    dzdxx = np.gradient(dzdx, axis=1)
    dzdyy = np.gradient(dzdy, axis=0)
    dzdxy = np.gradient(dzdx, axis=0)

    curvature = (dzdxx * (1 + dzdy ** 2) - 2 * dzdx * dzdy * dzdxy + dzdyy * (1 + dzdx ** 2)) / (
            (1 + dzdx ** 2 + dzdy ** 2) ** 1.5)
    return curvature

# 根据曲率检测异常值
def detect_anomalies(dem, curvature_lower_threshold, curvature_upper_threshold):
    curvature = compute_curvature(dem)

    # 更严格的异常值检测条件
    anomalies = np.zeros_like(dem, dtype=int)
    anomalies[(curvature > curvature_upper_threshold) | (curvature < curvature_lower_threshold)] = 1

    return anomalies

# 处理 DEM 并移除异常值
def process_dem_with_anomaly_detection(dem, curvature_lower_threshold, curvature_upper_threshold):
    dem_processed = dem.copy()
    anomalies = detect_anomalies(dem_processed, curvature_lower_threshold, curvature_upper_threshold)
    dem_processed[anomalies == 1] = np.nan
    return dem_processed

# 主函数
def imagedomain(d1, d2, lambdad, lambdax, mu1, au, bu, tau, tolerance):
    d1 = cp.array(d1, dtype=cp.float32)
    d2 = cp.array(d2, dtype=cp.float32)

    print(f'Initial d1 min: {cp.nanmin(d1)}, max: {cp.nanmax(d1)}')
    print(f'Initial d2 min: {cp.nanmin(d2)}, max: {cp.nanmax(d2)}')

    scale_factor = (d2.shape[0] / d1.shape[0], d2.shape[1] / d1.shape[1])
    d1_resampled = cp.array(zoom(cp.asnumpy(d1), scale_factor, order=1), dtype=cp.float32)

    mask1 = cp.isnan(d1_resampled)
    mask2 = cp.isnan(d2)
    combined_mask = mask1 | mask2

    d1_resampled[mask1] = 0
    d2[mask2] = 0

    w1 = cp.ones_like(d1_resampled)
    w2 = cp.ones_like(d2)

    w1[mask1] = 0
    w2[mask2] = 0

    N, _ = d2.shape
    max_iters = 1000
    mask = definemask(N)
    mask = cp.array(mask, dtype=cp.float32)

    Ffun = lambda x: cp.fft.fftn(x)
    Ftfun = lambda x: cp.fft.ifftn(x)

    D1 = cp.zeros((N, N))
    D1[1, 1] = 1
    D1[1, 0] = -1
    D2 = cp.zeros((N, N))
    D2[1, 1] = 1
    D2[0, 1] = -1

    d1FT = Ffun(D1)
    d2FT = Ffun(D2)
    DtD = cp.abs(d1FT) ** 2 + cp.abs(d2FT) ** 2

    alphainitmask = 2 * 1
    acceptpast = 10
    converged = False
    ii = 1

    x = cp.zeros_like(d2)
    lambda110 = cp.zeros_like(d2)
    lambda120 = cp.zeros_like(d2)
    x1 = cp.zeros_like(d2)
    x2 = cp.zeros_like(d2)
    Cu = cp.zeros_like(x)

    xprevious = x
    alphamask = alphainitmask
    objective = cp.zeros(max_iters + 1, dtype=cp.float32)
    F10 = mu1 * DtD + alphamask
    tprevious = 1
    yk = x

    total_start_time = time.time()
    start_time = time.time()

    while ii <= max_iters and not converged:
        Ax1 = cp.array(resample_to_low_res(cp.asnumpy(yk), 1 / np.array(scale_factor)))
        Ax2 = yk

        diff1 = cp.linalg.norm(Ax1 - d1_resampled)
        diff2 = cp.linalg.norm(Ax2 - d2)

        w1 = 2 * (1 / cp.log(1 + diff1)) / ((1 / cp.log(1 + diff1)) + (1 / cp.log(1 + diff2)))
        w2 = 2 * (1 / cp.log(1 + diff2)) / ((1 / cp.log(1 + diff1)) + (1 / cp.log(1 + diff2)))

        w1 = cp.ones_like(d1_resampled) * w1
        w2 = cp.ones_like(d2) * w2

        w1[mask1] = 0
        w2[mask2] = 0


        grad1 = w1 * (Ax1 - d1_resampled)
        grad2 = w2 * (Ax2 - d2)

        dx = xprevious
        z = yk - (grad1 + grad2) / alphamask

        ux = cp.pad(x[1:, :] - x[:-1, :], ((0, 1), (0, 0)), mode='constant')
        uy = cp.pad(x[:, 1:] - x[:, :-1], ((0, 0), (0, 1)), mode='constant')

        Unorm = cp.sqrt(ux ** 2 + uy ** 2)
        Unorm[Unorm == 0] = 1
        Cu = au + bu * (dxb(ux / Unorm) + dyb(uy / Unorm)) ** 2

        xx = dxf(x) - lambda110 / mu1
        xy = dyf(x) - lambda120 / mu1
        xf = cp.sqrt(xx ** 2 + xy ** 2)
        xf[xf == 0] = 1
        xf = cp.maximum(xf - Cu / mu1, 0) / xf
        x1 = xx * xf
        x2 = xy * xf

        g = alphamask * z - dxb(mu1 * x1 + lambda110) - dyb(mu1 * x2 + lambda120)

        x = cp.real(Ftfun(Ffun(g) / F10))

        lambda110 += mu1 * (x1 - dxf(x))
        lambda120 += mu1 * (x2 - dyf(x))

        t = (1 + cp.sqrt(1 + 4 * tprevious ** 2)) / 2
        yk = x + ((tprevious - 1) / t) * (x - xprevious)

        dx = x - dx
        normsqdx = cp.sum(dx ** 2)

        objective[ii] = computeobjectiveX(x, d1_resampled, d2, combined_mask, Cu, Ax1, Ax2, lambdax, lambdad, tau, w1,
                                          w2, 'gaussian', 1e-10, 'tv')

        tprevious = t
        xprevious = x
        converged = cp.abs(objective[ii] - objective[ii - 1]) / cp.abs(objective[ii]) <= tolerance

        if ii % 30 == 0:
            elapsed_time = time.time() - start_time
            print(f'Iteration: {ii}, Objective: {objective[ii]}')

        ii += 1

    optimization_time = time.time() - start_time
    print(f'Optimization Time: {optimization_time:.2f} seconds')

    reconstructed_DEM = x.get()

    # 检测和填补异常值
    reconstructed_DEM_before_anomalies = reconstructed_DEM.copy()
    curvature_lower_threshold = -0.2
    curvature_upper_threshold = 1
    max_anomaly_iterations = 1
    anomaly_tolerance = 20

    anomaly_start_time = time.time()
    anomaly_iteration = 0
    initial_anomalies = None
    final_anomalies = None
    initial_reconstructed = None
    final_reconstructed = None
    previous_num_anomalies = np.inf
    previous_reconstructed_DEM = None

    while anomaly_iteration < max_anomaly_iterations:
        anomalies = detect_anomalies(reconstructed_DEM, curvature_lower_threshold, curvature_upper_threshold)
        num_anomalies = np.sum(anomalies)
        if num_anomalies <= anomaly_tolerance or abs(previous_num_anomalies - num_anomalies) <= 1000 or num_anomalies > previous_num_anomalies:
            if abs(previous_num_anomalies - num_anomalies) <= 1000 or num_anomalies > previous_num_anomalies:
                reconstructed_DEM = previous_reconstructed_DEM
            break
        print(f'Iteration {anomaly_iteration}: Detected {num_anomalies} anomalies.')
        if anomaly_iteration == 0:
            initial_anomalies = anomalies.copy()
            initial_reconstructed = reconstructed_DEM.copy()
        previous_reconstructed_DEM = reconstructed_DEM.copy()
        reconstructed_DEM[anomalies == 1] = np.nan
        reconstructed_DEM = fill_nans_with_delta_surface_fill(reconstructed_DEM, anomalies)
        anomaly_iteration += 1
        previous_num_anomalies = num_anomalies
    final_anomalies = anomalies
    final_reconstructed = reconstructed_DEM.copy()

    anomaly_detection_time = time.time() - anomaly_start_time
    print(f'Anomaly Detection and Filling Time: {anomaly_detection_time:.2f} seconds')

    if np.isnan(reconstructed_DEM).any():
        print('Filling remaining NaNs with pre-anomaly-detection DEM.')
        reconstructed_DEM[np.isnan(reconstructed_DEM)] = reconstructed_DEM_before_anomalies[np.isnan(reconstructed_DEM)]

    total_time = time.time() - total_start_time
    print(f'Total Computation Time: {total_time:.2f} seconds')

    return reconstructed_DEM, ii, objective.get()


def computeobjectiveX(x, d1_resampled, d2, combined_mask, Cu, Ax1, Ax2, lambdax, lambdad, tau, w1, w2, noisetype,
                      logepsilon, penalty):
    mask_valid = ~combined_mask

    if noisetype == 'poisson':
        precompute1 = d1_resampled * cp.log(Ax1 + logepsilon)
        precompute2 = d2 * cp.log(Ax2 + logepsilon)
        objective = cp.sum(w1 * (cp.sum(Ax1[mask_valid]) - cp.sum(precompute1[mask_valid]))) + cp.sum(
            w2 * (cp.sum(Ax2[mask_valid]) - cp.sum(precompute2[mask_valid])))
    elif noisetype == 'gaussian':
        objective = cp.sum(w1 * cp.sum(((d1_resampled - Ax1) ** 2)[mask_valid]) / 2) + cp.sum(
            w2 * cp.sum(((d2 - Ax2) ** 2)[mask_valid]) / 2)
    elif noisetype == 'total':
        d1_masked = d1_resampled * mask
        d2_masked = d2 * mask
        objective = cp.sum(w1 * lambdax * cp.sum(((d1_resampled - Ax1) ** 2)[mask_valid]) / 2) + cp.sum(
            w2 * lambdad * cp.sum(((d2 - d2_masked) ** 2)[mask_valid]) / 2)

    if penalty == 'canonical':
        objective += cp.sum(cp.abs(tau * x))
    elif penalty == 'tv':
        x1 = dxf(x)
        x2 = dyf(x)
        objective += cp.sum(cp.abs(Cu * x1)) + cp.sum(cp.abs(Cu * x2))

    return objective


def main():
    y1_path = '/y1.tif'
    y2_path = '/y2.tif'

    with rasterio.open(y1_path) as src1:
        d1 = src1.read(1)
        profile1 = src1.profile

    with rasterio.open(y2_path) as src2:
        d2 = src2.read(1)
        profile2 = src2.profile

    d1[d1 > 3e+38] = np.nan
    d1[d1 < -3e+38] = np.nan

    curvature_lower_threshold = -2
    curvature_upper_threshold = 3
    d1_processed = process_dem_with_anomaly_detection(d1, curvature_lower_threshold, curvature_upper_threshold)

    d1 = d1_processed


    d1 = d1.astype(float)
    d2 = d2.astype(float)
    d1[d1 > 3e+38] = np.nan
    d1[d1 < -9997] = np.nan
    d2[d2 > 3e+38] = np.nan
    d2[d2 < -9997] = np.nan

    print(f'Loaded DEM1 shape: {d1.shape}')
    print(f'Loaded DEM1 min: {np.nanmin(d1)}, max: {np.nanmax(d1)}')
    print(f'Loaded DEM2 shape: {d2.shape}')
    print(f'Loaded DEM2 min: {np.nanmin(d2)}, max: {np.nanmax(d2)}')

    scale_factor = (d2.shape[0] / d1.shape[0], d2.shape[1] / d1.shape[1])
    d1_resampled = zoom(d1, scale_factor, order=1)

    d2_registered = register_images(d1_resampled, d2)

    lambdad = 0
    lambdax = 0
    mu1 = 0.001
    au = 2.0
    bu = 0.1
    tau = 0
    tolerance = 1e-5
    delta = 0

    reconstructed_DEM, ii, objective = imagedomain(
        d1_resampled, d2_registered, lambdad, lambdax, mu1, au, bu, tau, tolerance)

    print(f'Optimization finished in {ii} iterations')

    plot_results(d1, d2, reconstructed_DEM)

    # 保存重建后的DEM
    profile2.update(dtype=rasterio.float32, count=1, compress='lzw')
    with rasterio.open('/x_VFCR.tif', 'w', **profile2) as dst:
        dst.write(reconstructed_DEM.astype(rasterio.float32), 1)

    print('Reconstructed DEM saved as reconstructed_DEM.tif')


if __name__ == "__main__":
    main()
