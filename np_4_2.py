#!/usr/bin/env python
# coding: utf-8

import numpy as np  # linear algebra
import scipy.signal as sig
import scipy.fftpack as ffts
from pykalman import KalmanFilter
from scipy.linalg import hankel, svd
import pywt
import ewtpy
import emd


class statistic_:
    def __init__(self, data):
        # 二维数组，使用第一维分组
        self.data = np.array(data)

    def get_max(self):
        # 最大值
        if len(np.shape(self.data)) == 1:
            if isinstance(self.data[0], list):
                # 锯齿矩阵
                pool = []
                for v in self.data:
                    pool.append(max(v))
                return np.array(pool)
            else:
                # 向量
                return max(self.data)
        else:
            # 矩形矩阵
            return self.data.max(-1)

    def get_min(self):
        # 最小值
        if len(np.shape(self.data)) == 1:
            if isinstance(self.data[0], list):
                # 锯齿矩阵
                pool = []
                for v in self.data:
                    pool.append(min(v))
                return np.array(pool)
            else:
                # 向量
                return min(self.data)
        else:
            # 矩形矩阵
            return self.data.min(-1)

    def get_avg(self):
        # 均值
        if len(np.shape(self.data)) == 1:
            if isinstance(self.data[0], list):
                # 锯齿矩阵
                pool = []
                for v in self.data:
                    pool.append(np.mean(v))
                return np.array(pool)
            else:
                # 向量
                return np.mean(self.data)
        else:
            # 矩形矩阵
            return np.mean(self.data, axis=-1)

    def get_mid(self):
        # 中位数
        if len(np.shape(self.data)) == 1:
            if isinstance(self.data[0], list):
                # 锯齿矩阵
                pool = []
                for v in self.data:
                    pool.append(np.median(v))
                return np.array(pool)
            else:
                # 向量
                return np.median(self.data)
        else:
            # 矩形矩阵
            return np.median(self.data, axis=-1)

    def get_len(self):
        # 计数
        # 返回 tuple 存储矩形矩阵尺寸， numpy.ndarray 存储锯齿矩阵每组数据长度
        if len(np.shape(self.data)) == 1:
            if isinstance(self.data[0], list):
                # 锯齿矩阵
                pool = []
                for v in self.data:
                    pool.append(len(v))
                return np.array(pool)
            else:
                # 向量
                return np.shape(self.data)
        else:
            # 矩形矩阵
            return np.shape(self.data)

    def get_sum(self):
        # 求和
        if len(np.shape(self.data)) == 1:
            if isinstance(self.data[0], list):
                # 锯齿矩阵
                pool = []
                for v in self.data:
                    pool.append(sum(v))
                return np.array(pool)
            else:
                # 向量
                return np.sum(self.data)
        else:
            # 矩形矩阵
            return np.sum(self.data, axis=-1)

    def get_cup(self):
        # 组数据求积
        if len(np.shape(self.data)) == 1:
            if isinstance(self.data[0], list):
                # 锯齿矩阵
                pool = []
                for v in self.data:
                    pool.append(np.cumprod(v)[-1])
                return np.array(pool)
            else:
                # 向量
                return np.cumprod(self.data)[-1]
        else:
            # 矩形矩阵
            return np.cumprod(self.data, axis=-1)[:, -1]

    def get_var(self):
        # 方差
        if len(np.shape(self.data)) == 1:
            if isinstance(self.data[0], list):
                # 锯齿矩阵
                pool = []
                for v in self.data:
                    pool.append(np.var(v))
                return np.array(pool)
            else:
                # 向量
                return np.var(self.data)
        else:
            # 矩形矩阵
            return np.var(self.data, axis=-1)

    def get_std(self):
        # 标准差
        return np.sqrt(self.get_var())

    def get_mnt(self, order=3):
        # 3阶以上标准中心距
        # order 阶次，取自然数
        # 3阶 偏度
        # 4阶 峰度（峭度）
        # 5阶 超偏度
        # 6阶 超尾度

        data_avg = self.get_avg()
        data_std = self.get_std()
        data_len = self.get_len()  # 锯齿数组和矩形数组 data_len 的格式不同

        if order == 0:
            return self.get_sum()
        elif order == 1:
            return data_avg
        elif order == 2:
            return data_std**2
        else:
            if len(np.shape(self.data)) == 1:
                if isinstance(self.data[0], list):
                    # 锯齿矩阵
                    pool = []
                    for i, v in enumerate(self.data):
                        pool.append(
                            sum((v - data_avg[i]) ** order)
                            / (data_len[i] * data_std[i] ** order)
                        )
                    return np.array(pool)
                else:
                    # 向量
                    return sum((self.data - data_avg) ** order) / data_len
            else:
                # 矩形矩阵
                pool = []
                for i, v in enumerate(self.data):
                    # 锯齿数组和矩形数组 data_len 的格式不同
                    pool.append(
                        sum((v - data_avg[i]) ** order)
                        / (data_len[1] * data_std[i] ** order)
                    )
                return np.array(pool)

    def get_pp(self):
        # 峰峰值
        if len(np.shape(self.data)) == 1:
            if isinstance(self.data[0], list):
                # 锯齿矩阵
                pool = []
                for v in self.data:
                    pool.append(max(v) - min(v))
                return np.array(pool)
            else:
                # 向量
                return np.max(self.data) - np.min(self.data)
        else:
            # 矩形矩阵
            return np.max(self.data, axis=-1) - np.min(self.data, axis=-1)

    def get_ppc(self):
        # （皮尔逊）相关系数
        if len(np.shape(self.data)) == 1:
            if isinstance(self.data[0], list):
                # 锯齿矩阵
                # 变量数不相等，不具有一一对应关系
                return None
        else:
            return np.corrcoef(self.data)

    def get_qtl(self, interval=10):
        # 计算分位点
        # interval 为区间数
        data_len = self.get_len()  # 锯齿数组和矩形数组 data_len 的格式不同

        if len(np.shape(self.data)) == 1:
            if isinstance(self.data[0], list):
                # 锯齿矩阵
                pool = []
                for i, v in enumerate(self.data):
                    # 锯齿数组和矩形数组 data_len 的格式不同
                    quantiles = np.linspace(0, 1, interval + 1) * data_len[i]
                    quantiles[-1] = -1
                    quantiles = quantiles.astype(np.int64)
                    pool.append(np.sort(np.array(v))[quantiles])
                return np.array(pool)
            else:
                # 向量
                quantiles = np.linspace(0, 1, interval + 1) * data_len
                quantiles[-1] = -1
                quantiles = quantiles.astype(np.int64)
                return np.sort(self.data)[quantiles]
        else:
            # 矩形矩阵
            pool = []
            for v in self.data:
                # 锯齿数组和矩形数组 data_len 的格式不同
                quantiles = np.linspace(0, 1, interval + 1) * data_len[1]
                quantiles[-1] = -1
                quantiles = quantiles.astype(np.int64)
                pool.append(np.sort(v)[quantiles])
            return np.array(pool)


class time_dom_:
    def __init__(self, data):
        # 二维数组，使用第一维分组
        self.data = np.array(data)
        # 锯齿数组和矩形数组 data_len 的格式不同
        self.len = statistic_(self.data).get_len()
        # 判断数据形状
        if len(np.shape(self.data)) == 1:
            if isinstance(self.data[0], list):
                # 锯齿矩阵
                self.mtx_type = 2
            else:
                # 向量
                self.mtx_type = 0
        else:
            # 矩形矩阵
            self.mtx_type = 1

    def mul_win(self, win_name="hamming", std=0.65):
        # 加窗分析
        # win_name 为窗类型 \
        # 包含 'hamming', 'hann', 'triang', 'boxcar', 'gaussian'等 \
        # 对应 汉明窗，汉宁窗，三角窗，矩形窗，高斯窗等
        # 调用 高斯窗 时需要传递方差 std 参数

        if self.mtx_type == 0:
            # 向量
            if win_name == "gaussian":
                y = sig.get_window((win_name, std), self.len[0])
            else:
                y = sig.get_window(win_name, self.len[0])
            return np.multiply(self.data, y)

        if self.mtx_type == 1:
            # 矩形矩阵
            if win_name == "gaussian":
                y = sig.get_window((win_name, std), self.len[1])
            else:
                y = sig.get_window(win_name, self.len[1])
            return np.multiply(self.data, y)

        if self.mtx_type == 2:
            # 锯齿矩阵
            pool = []
            for i, v in enumerate(self.data):
                if win_name == "gaussian":
                    y = sig.get_window((win_name, std), self.len[i])
                else:
                    y = sig.get_window(win_name, self.len[i])
                pool.append(np.multiply(self.data[i], y))
            return np.array(pool, dtype=object)

    def spt_frm(self, fame_len, overlap_rate=0, drop_last=1):
        # 信号分帧
        # fame_len 为每帧长度
        # overlap_rate 为重叠率，代表每帧与前一帧重叠度，小于1
        # drop_last 为是否补0保留真实长度不够一帧长度的最后一批数据
        if self.mtx_type == 0:
            # 向量
            start = 0
            end = start + fame_len
            pool = []
            while start < self.len[0]:
                if end <= self.len[0]:
                    pool.append(self.data[start:end])
                else:
                    if drop_last:
                        pool.append(
                            np.pad(
                                self.data[start:],
                                (0, fame_len - len(self.data[start:])),
                            )
                        )
                        break
                    else:
                        break
                start = start + int(np.ceil(fame_len * (1 - overlap_rate)))
                end = start + fame_len
            return np.array(pool)

        if self.mtx_type == 1:
            # 矩形矩阵
            pool_1 = []
            for v in self.data:
                start = 0
                end = start + fame_len
                pool = []
                while start < self.len[1]:
                    if end <= self.len[1]:
                        pool.append(v[start:end])
                    else:
                        if drop_last:
                            pool.append(
                                np.pad(v[start:], (0, fame_len - len(v[start:])))
                            )
                            break
                        else:
                            break
                    start = start + int(np.ceil(fame_len * (1 - overlap_rate)))
                    end = start + fame_len
                pool_1.append(pool)
            return np.array(pool_1)

        if self.mtx_type == 2:
            # 锯齿矩阵
            pool_1 = []
            # 锯齿数组和矩形数组 data_len 的格式不同
            for i, v in enumerate(self.data):
                start = 0
                end = start + fame_len
                pool = []
                while start < self.len[i]:
                    if end <= self.len[i]:
                        pool.append(v[start:end])
                    else:
                        if drop_last:
                            pool.append(
                                np.pad(v[start:], (0, fame_len - len(v[start:])))
                            )
                            break
                        else:
                            break
                    start = start + int(np.ceil(fame_len * (1 - overlap_rate)))
                    end = start + fame_len
                pool_1.append(pool)
            return np.array(pool_1)

    def sld_avg(self, scale=4):
        # 滑动平均
        if self.mtx_type == 0:
            # 向量
            start = 0
            end = start + scale
            pool = []
            while start < self.len[0]:
                if end <= self.len[0]:
                    pool.append(np.mean(self.data[start:end]))
                else:
                    break
                start = start + 1
                end = start + scale
            return np.array(pool)

        if self.mtx_type == 1:
            # 矩形矩阵
            pool_1 = []
            for v in self.data:
                start = 0
                end = start + scale
                pool = []
                while start < self.len[1]:
                    if end <= self.len[1]:
                        pool.append(np.mean(v[start:end]))
                    else:
                        break
                    start = start + 1
                    end = start + scale
                pool_1.append(pool)
            return np.array(pool_1)

        if self.mtx_type == 2:
            # 锯齿矩阵
            pool_1 = []
            # 锯齿数组和矩形数组 data_len 的格式不同
            for i, v in enumerate(self.data):
                start = 0
                end = start + scale
                pool = []
                while start < self.len[i]:
                    if end <= self.len[i]:
                        pool.append(np.mean(v[start:end]))
                    else:
                        break
                    start = start + 1
                    end = start + scale
                pool_1.append(pool)
            return np.array(pool_1)

    def get_cov(self, cov_with, mode="same"):
        # 卷积
        # mode 指定输出方式 \
        # 'full’完全离散线性卷积 \
        # 'valid’输出仅包含那些不依赖于零填充的元素 \
        # 'same’输出与in1的大小相同，以‘full’输出为中心

        if self.mtx_type == 0:
            # 向量
            return sig.convolve(in1=self.data, in2=cov_with, mode=mode)

        if self.mtx_type in (1, 2):
            # 矩形矩阵，锯齿矩阵
            pool = []
            for v in self.data:
                pool.append(sig.convolve(in1=v, in2=cov_with, mode=mode))
            return np.array(pool)

    def up_samp(self, ins_rate=1, mode="zero"):
        # 升采样
        # ins_rate 增加的采样倍数
        # mode 插值方式，'zero' 零插值，'linear' 线性插值，'ffill' 用前一个值填充，'bfill' 用后一个值填充

        if self.mtx_type == 0:
            # 向量
            pool = []
            for i in range(self.len[0] - 1):
                pool.append(self.data[i])
                if mode == "zero":
                    pool.extend(np.zeros(ins_rate))
                    continue
                if mode == "linear":
                    pool.extend(
                        np.linspace(self.data[i], self.data[i + 1], ins_rate + 2)[1:-1]
                    )
                    continue
                if mode == "ffill":
                    pool.extend(np.ones(ins_rate) * self.data[i])
                    continue
                if mode == "bfill":
                    pool.extend(np.ones(ins_rate) * self.data[i + 1])
            pool.append(self.data[-1])
            return np.array(pool)

        if self.mtx_type == 1:
            # 矩形矩阵
            pool_1 = []
            for v in self.data:
                pool = []
                for i in range(self.len[1] - 1):
                    pool.append(v[i])
                    if mode == "zero":
                        pool.extend(np.zeros(ins_rate))
                        continue
                    if mode == "linear":
                        pool.extend(np.linspace(v[i], v[i + 1], ins_rate + 2)[1:-1])
                        continue
                    if mode == "ffill":
                        pool.extend(np.ones(ins_rate) * v[i])
                        continue
                    if mode == "bfill":
                        pool.extend(np.ones(ins_rate) * v[i + 1])
                pool.append(v[-1])
                pool_1.append(pool)
            return np.array(pool_1)

        if self.mtx_type == 2:
            # 锯齿矩阵
            pool_1 = []
            # 锯齿数组和矩形数组 data_len 的格式不同
            for i, v in enumerate(self.data):
                pool = []
                for i in range(self.len[i] - 1):
                    pool.append(v[i])
                    if mode == "zero":
                        pool.extend(np.zeros(ins_rate))
                        continue
                    if mode == "linear":
                        pool.extend(np.linspace(v[i], v[i + 1], ins_rate + 2)[1:-1])
                        continue
                    if mode == "ffill":
                        pool.extend(np.ones(ins_rate) * v[i])
                        continue
                    if mode == "bfill":
                        pool.extend(np.ones(ins_rate) * v[i + 1])
                pool.append(v[-1])
                pool_1.append(pool)
            return np.array(pool_1)

    def down_samp(self, gap=1):
        # 降采样
        if self.mtx_type == 0:
            # 向量
            idx = np.arange(0, self.len[0], gap + 1)
            return self.data[idx]

        if self.mtx_type == 1:
            # 矩形矩阵
            idx = np.arange(0, self.len[1], gap + 1)
            return self.data[:, idx]

        if self.mtx_type == 2:
            # 锯齿矩阵
            pool = []
            # 锯齿数组和矩形数组 data_len 的格式不同
            for i, v in enumerate(self.data):
                idx = np.arange(0, self.len[i], gap + 1)
                pool.append(np.array(v)[idx])
            return np.array(pool)

    def crs_crr(self, target):
        """交叉关联分析（相关度分析）
        输入：
            target 为比较的目标波形
        输出：
            corr 为与目标波形的交叉相关系数
            lags 为 corr 与目标波形的偏移度
        """
        if self.mtx_type == 0:
            # 向量
            corr = sig.correlate(self.data, target)
            lags = sig.correlation_lags(len(target), self.len[0])
            return [corr.tolist(), lags.tolist()]

        if self.mtx_type == 1:
            # 矩形矩阵
            corr = sig.correlate(self.data, np.tile(target, (self.len[0], 1)))
            lags = sig.correlation_lags(len(target), self.len[1])
            return [corr.tolist(), lags.tolist()]

        if self.mtx_type == 2:
            # 锯齿矩阵
            # 锯齿数组和矩形数组 data_len 的格式不同
            corr_pool = []
            lags_pool = []
            for i, v in enumerate(self.data):
                corr = sig.correlate(v, target)
                lags = sig.correlation_lags(len(target), self.len[i])
                corr_pool.append(corr)
                lags_pool.append(lags)
            return [corr_pool.tolist(), lags_pool.tolist()]


class freq_dom_:
    def __init__(self, data):
        self.data = np.array(data)
        # 锯齿数组和矩形数组 data_len 的格式不同
        self.len = statistic_(self.data).get_len()
        # 判断数据形状
        if len(np.shape(self.data)) == 1:
            if isinstance(self.data[0], list):
                # 锯齿矩阵
                self.mtx_type = 2
            else:
                # 向量
                self.mtx_type = 0
        else:
            # 矩形矩阵
            self.mtx_type = 1

    def get_dct(self):
        # DCT，离散余弦变换，DCT-II公式
        if self.mtx_type in (0, 1):
            # 向量和矩形矩阵
            return ffts.dct(self.data)
        if self.mtx_type == 2:
            # 锯齿矩阵
            pool = []
            for v in self.data:
                pool.append(ffts.dct(v))
            return np.array(pool, dtype=object)

    def get_gabor(self, fs, std=0.65, nperseg=256, noverlap=None):
        """gabor变换（高斯窗短时傅里叶）
        输入：
            std 高斯窗标准差
            fs 采样频率
            nperseg 每窗采样点数
            noverlap 每窗重叠点数
        输出：
            f 频率分辨坐标
            t 每窗时间坐标
            Zxx.real 对应每窗stft变换实值
        """
        if self.mtx_type in (0, 1):
            # 向量
            f, t, Zxx = sig.stft(
                self.data,
                fs=fs,
                window=("gaussian", std),
                nperseg=nperseg,
                noverlap=noverlap,
            )
            return [f.tolist(), t.tolist(), Zxx.real.tolist()]

        if self.mtx_type == 2:
            # 锯齿矩阵
            f_pool, t_pool, Zxx_pool = [], [], []
            for v in self.data:
                f, t, Zxx = sig.stft(
                    np.array(v),
                    fs=fs,
                    window=("gaussian", std),
                    nperseg=nperseg,
                    noverlap=noverlap,
                )
                f_pool.append(list(f))
                t_pool.append(list(t))
                Zxx_pool.append(list(Zxx.real))
            return [f_pool, t_pool, Zxx_pool]

    def dig_filt(self, filt_type, fs, fc, filter_od=8):
        """数字滤波
        输入：
            filt_type 滤波类型 ‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’
            fs 采样频率
            fc 截止频率，可为 单值 和 二值数组
            filter_od 滤波器阶数
        """
        if self.mtx_type in (0, 1):
            # 向量，矩形矩阵
            wn = 2 * np.array(fc) / fs
            b, a = sig.butter(filter_od, wn, filt_type)
            return sig.filtfilt(b, a, self.data)

        if self.mtx_type == 2:
            # 锯齿矩阵
            wn = 2 * np.array(fc) / fs
            b, a = sig.butter(filter_od, wn, filt_type)
            pool = []
            for v in self.data:
                pool.append(list(sig.filtfilt(b, a, v)))
            return np.array(pool, dtype=object)

    def wie_filt(self):
        # 维纳滤波
        if self.mtx_type in (0, 1):
            # 向量，矩形矩阵
            return sig.wiener(self.data)
        if self.mtx_type == 2:
            # 锯齿矩阵
            pool = []
            for v in self.data:
                pool.append(list(sig.wiener(v)))
            return np.array(pool, dtype=object)

    def kal_filt(self):
        # 卡尔曼滤波
        kf = KalmanFilter(n_dim_obs=1)
        if self.mtx_type == 0:
            # 向量
            return np.array(kf.em(self.data).smooth(self.data)[0]).flatten()
        if self.mtx_type in (1, 2):
            # 矩形矩阵和锯齿矩阵
            pool = []
            for v in self.data:
                pool.append(np.array(kf.em(v).smooth(v)[0]).flatten())
            return pool

    def LMS(self, xn, dn, M, mu):
        """单通道自适应滤波 （Least Mean Square，LMS）
        输入：
            xn: 原数据
            dn: 目标波
            M: 滤波器阶数
            mu: 步长因子
        输出：
            yn: 结果波
            en: 结果波与目标波误差
        """
        L = len(xn)  # 采样点数
        en = np.zeros(L)
        W = np.zeros((L, M))  # 权重矩阵
        for k in range(M - 1, L):
            x = np.array(xn[k - M + 1 : k + 1][::-1])  # 逆序
            d = np.array(dn[k])
            y = np.multiply(W[k - 1], x).sum()  # 加权求和滤波
            en[k] = d - y
            W[k] = np.add(W[k - 1], 2 * mu * en[k] * x)  # 求导更新权重

        # 求最优时滤波器的输出序列
        yn = np.inf * np.ones(len(xn))
        for k in range(M - 1, len(xn)):
            x = np.array(xn[k - M + 1 : k + 1][::-1])
            yn[k] = np.multiply(W[k], x).sum()

        return yn, en

    def lms_filt(self, dn, M=8, mu=1e-3):
        """自适应滤波 （Least Mean Square，LMS）
        输入：
            dn: 目标波
            M: 滤波器阶数
            mu: 步长因子
        输出：
            yn: 结果波
            en: 结果波与目标波误差
        """
        if self.mtx_type == 0:
            # 向量
            yn, en = self.LMS(self.data, dn, M, mu)
            return [yn.tolist(), en.tolist()]
        if self.mtx_type in (1, 2):
            # 矩形矩阵和锯齿矩阵
            yn_pool = []
            en_pool = []
            for v in self.data:
                yn, en = self.LMS(v, dn, M, mu)
                yn_pool.append(yn.tolist())
                en_pool.append(en.tolist())
            return [yn_pool, en_pool]

    def FFT_HARM(self, data, fs, cut, flag_xd2pi):
        """单通道傅里叶谐波分析
        输入：
            fs: 采样频率
            cut: 主要波强度阈值，0 < cut <= 1
        输出：
            mf_x: 主要波频率
            mf_y: 主要波强度
            fft_x: 傅里叶变换频率轴
            fft_y: 频率强度
        """
        N = len(data)
        T = N / fs
        y = data

        y = y - np.mean(y)

        fft_y = np.abs(ffts.fft(np.array(y)))[: int(N / 2)]
        fft_y /= max(fft_y)

        if flag_xd2pi:
            # 直接间隔有倍率 2*np.pi
            fft_x = ffts.fftfreq(N, T / N / (2 * np.pi))[: int(N / 2)]
        else:
            fft_x = ffts.fftfreq(N, T / N)[: int(N / 2)]

        # 奈奎斯特定理
        fft_y = fft_y[: sum(fft_x < fs / 2.56)]
        fft_x = fft_x[: sum(fft_x < fs / 2.56)]

        fft_y_sort_idx = np.argsort(fft_y)
        idx = fft_y_sort_idx[sum(fft_y <= cut) :]
        mf_x = fft_x[idx[::-1]]
        mf_y = fft_y[idx[::-1]]

        return mf_x, mf_y, fft_x, fft_y

    def get_harm(self, fs, cut=0.3, flag_xd2pi=0):
        """傅里叶谐波分析
        输入：
            fs: 采样频率
            cut: 主要波强度阈值，0 < cut <= 1
            flag_xd2pi: 是否把频率坐标轴缩小2pi倍
        输出：
            mf_x: 主要波频率
            mf_y: 主要波强度
        """
        if self.mtx_type == 0:
            # 向量
            mf_x, mf_y, _, _ = self.FFT_HARM(self.data, fs, cut, flag_xd2pi)
            return [mf_x.tolist(), mf_y.tolist()]
        if self.mtx_type in (1, 2):
            # 矩形矩阵和锯齿矩阵
            mf_x_pool = []
            mf_y_pool = []
            for v in self.data:
                mf_x, mf_y, _, _ = self.FFT_HARM(v, fs, cut, flag_xd2pi)
                mf_x_pool.append(mf_x.tolist())
                mf_y_pool.append(mf_y.tolist())
            return [mf_x_pool, mf_y_pool]

    def SVD_DENOI(self, data, ratio):
        """SVD降噪
        输入：
            ratio：保留的奇异值阶数占数据长度的百分比
        """
        H = hankel(data)  # 汉克尔矩阵 (Hankel Matrix)
        U, S, V = svd(H)
        S[round(len(S) * ratio) :] = 0
        H_new = U * np.mat(np.diag(S)) * V
        return np.array(H_new[0, :])[0]

    def denoise(self, method="svd", ratio=0.01, kernel_size=5):
        """降噪，提供两种方式，中值滤波降噪 和 SVD降噪
        输入：
            method：降噪方式，'med'或者'svd'，对应 中值滤波降噪 和 SVD降噪
            ratio：'svd'降噪保留的奇异值阶数占数据长度的百分比
            kernel_size：'med'降噪的滤波器长度
        """
        if self.mtx_type == 0:
            if method == "med":
                return sig.medfilt(self.data, kernel_size)
            else:
                return self.SVD_DENOI(self.data, ratio)
        if self.mtx_type in (1, 2):
            pool = []
            for v in self.data:
                if method == "med":
                    pool.append(sig.medfilt(v, kernel_size))
                else:
                    pool.append(self.SVD_DENOI(v, ratio))
            return np.array(pool)

    def PSD(self, data, fs, method, window, nperseg, noverlap, flag_xd2pi):
        """单通道功率谱（Power Spectral Density）计算，提供两种方式 周期图法 和 多窗谱法
        输入：
            method：计算方式，'periodogram'或者'welch'，对应 周期图法 和 多窗谱法
            window：多窗谱法所用窗，可用包含加窗计算算法内的所有窗类型
            nperseg：多窗谱法所用窗长度
            noverlap：多窗谱法所每窗重叠点数
        输出：
            f_new：频率轴
            Pxx_new：信号功率谱密度
        """
        if method == "periodogram":
            f, Pxx = sig.periodogram(data, fs)
        elif method == "welch":
            f, Pxx = sig.welch(
                data, fs, window=window, nperseg=nperseg, noverlap=noverlap
            )
        else:
            return None

        # 直接间隔有倍率 2*np.pi
        if flag_xd2pi:
            f = f * 2 * np.pi
        # 奈奎斯特定理
        f_new = f[: sum(f < fs / 2.56)]
        Pxx_new = Pxx[: sum(f < fs / 2.56)]

        return f_new, Pxx_new

    def get_psd(
        self, fs, method="periodogram", window="hann", nperseg=256, noverlap=None, flag_xd2pi=0
    ):
        """功率谱计算，提供两种方式 周期图法 和 多窗谱法
        输入：
            method：计算方式，'periodogram'或者'welch'，对应 周期图法 和 多窗谱法
            window：多窗谱法所用窗，可用包含加窗计算算法内的所有窗类型
            nperseg：多窗谱法所用窗长度
            noverlap：多窗谱法所每窗重叠点数
        输出：
            f：频率轴
            Pxx：信号功率谱密度
        """
        if self.mtx_type == 0:
            f, Pxx = self.PSD(self.data, fs, method, window, nperseg, noverlap, flag_xd2pi)
            return [f.tolist(), Pxx.tolist()]
        if self.mtx_type in (1, 2):
            pool_f = []
            pool_Pxx = []
            for v in self.data:
                f, Pxx = self.PSD(v, fs, method, window, nperseg, noverlap, flag_xd2pi)
                pool_f.append(f.tolist())
                pool_Pxx.append(Pxx.tolist())
            return [pool_f, pool_Pxx]


class time_freq_dom_:
    def __init__(self, data):
        self.data = np.array(data)
        # 锯齿数组和矩形数组 data_len 的格式不同
        self.len = statistic_(self.data).get_len()
        # 判断数据形状
        if len(np.shape(self.data)) == 1:
            if isinstance(self.data[0], list):
                # 锯齿矩阵
                self.mtx_type = 2
            else:
                # 向量
                self.mtx_type = 0
        else:
            # 矩形矩阵
            self.mtx_type = 1

    def WAVPACK(self, data, wavelet, level, mode, order):
        """单通道信号小波包分析
            输入：
                wavlet：母小波 \
                    haar : Haar 家族，含 ['haar']
                    db : Daubechies 家族，含 ['db1', 'db2', 'db3', '...']
                    sym : Symlets 家族，含 ['sym2', 'sym3', 'sym4', '...']
                    coif : Coiflets 家族，含 ['coif1', 'coif2', 'coif3', '...']
                    bior : Biorthogonal 家族，含 ['bior1.1', 'bior1.3', 'bior1.5', '...']
                    rbio : Reverse biorthogonal 家族，含 ['rbio1.1', 'rbio1.3', 'rbio1.5', '...']
                    dmey : Discrete Meyer (FIR Approximation) 家族，含 ['dmey']
                    gaus : Gaussian 家族，含 ['gaus1', 'gaus2', 'gaus3', '...']
                    mexh : Mexican hat wavelet 家族，含 ['mexh']
                    morl : Morlet wavelet 家族，含 ['morl']
                    cgau : Complex Gaussian wavelets 家族，含 ['cgau1', 'cgau2', 'cgau3', '...']
                    shan : Shannon wavelets 家族，含 ['shan']
                    fbsp : Frequency B-Spline wavelets 家族，含 ['fbsp']
                    cmor : Complex Morlet wavelets 家族，含 ['cmor']
                level：分析层级，整数，字符'max'表示返回最大层级结果（不推荐）
                mode：信号填充方式，'zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization', 'reflect', 'antisymmetric', 'antireflect'之一
                order：小波包树节点排列方式，'natural'或者'freq'，分别对应 树节点顺序 和 频率顺序 
            输出：
                rec_results：单节点小波系数重构结果
                coeffs：各节点小波系数，排列方式对应输入的order参数
        """
        wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode=mode)
        if level == "max":
            level = wp.maxlevel
        elif level > wp.maxlevel:
            level = wp.maxlevel
        node_name_list = [node.path for node in wp.get_level(level=level, order=order)]

        rec_results = []
        coeffs = []
        for i in node_name_list:
            new_wp = pywt.WaveletPacket(
                data=np.zeros(len(data)), wavelet=wavelet, mode=mode
            )
            new_wp[i] = wp[i].data
            x_i = new_wp.reconstruct(update=True)
            rec_results.append(x_i)
            coeffs.append(wp[i].data)

        return np.array(rec_results), np.array(coeffs)

    def get_wavpack(self, wavelet="db4", level=3, mode="symmetric", order="freq"):
        """小波包分析
            输入：
                wavlet：母小波 \
                    haar : Haar 家族，含 ['haar']
                    db : Daubechies 家族，含 ['db1', 'db2', 'db3', '...']
                    sym : Symlets 家族，含 ['sym2', 'sym3', 'sym4', '...']
                    coif : Coiflets 家族，含 ['coif1', 'coif2', 'coif3', '...']
                    bior : Biorthogonal 家族，含 ['bior1.1', 'bior1.3', 'bior1.5', '...']
                    rbio : Reverse biorthogonal 家族，含 ['rbio1.1', 'rbio1.3', 'rbio1.5', '...']
                    dmey : Discrete Meyer (FIR Approximation) 家族，含 ['dmey']
                    gaus : Gaussian 家族，含 ['gaus1', 'gaus2', 'gaus3', '...']
                    mexh : Mexican hat wavelet 家族，含 ['mexh']
                    morl : Morlet wavelet 家族，含 ['morl']
                    cgau : Complex Gaussian wavelets 家族，含 ['cgau1', 'cgau2', 'cgau3', '...']
                    shan : Shannon wavelets 家族，含 ['shan']
                    fbsp : Frequency B-Spline wavelets 家族，含 ['fbsp']
                    cmor : Complex Morlet wavelets 家族，含 ['cmor']
                level：分析层级，整数，字符'max'表示返回最大层级结果（不推荐）
                mode：信号填充方式，'zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization', 'reflect', 'antisymmetric', 'antireflect'之一
                order：小波包树节点排列方式，'natural'或者'freq'，分别对应 树节点顺序 和 频率顺序 
            输出：
                rec_results：单节点小波系数重构结果
                coeffs：各节点小波系数，排列方式对应输入的order参数
        """
        if self.mtx_type == 0:
            rec_results, coeffs = self.WAVPACK(self.data, wavelet, level, mode, order)
            return [rec_results.tolist(), coeffs.tolist()]
        if self.mtx_type in (1, 2):
            pool_rec_results = []
            pool_coeffs = []
            for v in self.data:
                rec_results, coeffs = self.WAVPACK(v, wavelet, level, mode, order)
                pool_rec_results.append(rec_results.tolist())
                pool_coeffs.append(coeffs.tolist())
            return [pool_rec_results, pool_coeffs]

    def get_ewt(
        self,
        N=5,
        log=0,
        detect="locmax",
        completion=0,
        reg="none",
        lengthFilter=10,
        sigmaFilter=1,
    ):
        """经验小波分析
        输入：
            N：信号分量数
            log：是否采用对数频率，0或1
            detect：区间划分方式，'locmax','locmaxmin','locmaxminf'之一，对应 最大值之间的中点、最大值之间的最小值、原始谱最大值之间的最小值
            completion：模式划分数量不足 N 时是否完成，0或1
            reg：滤波方式，'none','gaussian','average'之一，对应 无、高斯滤波、平均滤波
            lengthFilter：滤波长度
            sigmaFilter：高斯滤波标准差
        输出：
            ewt：信号分量
            mfb：滤波器组（频域）
            boundaries：标准化至pi的划分边界
        """
        if self.mtx_type == 0:
            ewt, mfb, boundaries = ewtpy.EWT1D(
                self.data, N, log, detect, completion, reg, lengthFilter, sigmaFilter
            )
            return [ewt.T.tolist(), mfb.T.tolist(), boundaries.tolist()]
        if self.mtx_type in (1, 2):
            pool_ewt = []
            pool_mfb = []
            pool_boundaries = []
            for v in self.data:
                ewt, mfb, boundaries = ewtpy.EWT1D(
                    np.array(v),
                    N,
                    log,
                    detect,
                    completion,
                    reg,
                    lengthFilter,
                    sigmaFilter,
                )
                pool_ewt.append(list(ewt.T))
                pool_mfb.append(list(mfb.T))
                pool_boundaries.append(list(boundaries))
            return [pool_ewt, pool_mfb, pool_boundaries]

    def HAAR_LWT(self, x):
        # TODO
        N = len(x)
        s = np.zeros(N // 2)
        d = np.zeros(N // 2)
        for i in range(N // 2):
            s[i] = (x[2 * i] + x[2 * i + 1]) / 2
            d[i] = x[2 * i] - s[i]
        return s, d

    def get_lwt(self):
        # TODO，提升小波分析
        if self.mtx_type == 0:
            s, d = self.HAAR_LWT(self.data)
            return [s.tolist(), d.tolist()]
        if self.mtx_type in (1, 2):
            pool_s = []
            pool_d = []
            for v in self.data:
                s, d = self.HAAR_LWT(v)
                pool_s.append(s.tolist())
                pool_d.append(d.tolist())
            return [pool_s, pool_d]

    def get_emd(self):
        # 经验模态分解（EMD）
        if self.mtx_type == 0:
            return emd.sift.sift(self.data).T
        if self.mtx_type in (1, 2):
            pool = []
            for v in self.data:
                pool.append(emd.sift.sift(np.array(v)).T)
            return np.array(pool)

    def get_hht(self, fs, intervals=100):
        """Hilbert谱 和 边际谱（Marginal Spectrum）
        输入：
            fs：采样频率
            intervals：频率轴划分区间数
        输出：
            f：频率坐标
            hht：每个时间点对应所有频率坐标变换结果
            MS：边际谱
        """
        if self.mtx_type == 0:
            imf = self.get_emd().T
            IP, IF, IA = emd.spectra.frequency_transform(imf, fs, "hilbert")
            freq_range = (
                0,
                fs / 2.56,
                intervals,
            )  # 0 to fs/2.56Hz in (intervals) steps
            f, hht = emd.spectra.hilberthuang(IF, IA, freq_range, sum_time=False)
            MS = np.sum(hht, 1)
            return [f.tolist(), hht.tolist(), MS.tolist()]
        if self.mtx_type in (1, 2):
            imf = self.get_emd()
            f_pool = []
            hht_pool = []
            MS_pool = []
            for mtx in imf:
                mtx = mtx.T
                IP, IF, IA = emd.spectra.frequency_transform(mtx, fs, "hilbert")
                freq_range = (
                    0,
                    fs / 2.56,
                    intervals,
                )  # 0 to fs/2.56Hz in (intervals) steps
                f, hht = emd.spectra.hilberthuang(IF, IA, freq_range, sum_time=False)
                f_pool.append(f.tolist())
                hht_pool.append(hht.tolist())
                MS_pool.append(np.sum(hht, 1).tolist())
            return [f_pool, hht_pool, MS_pool]

    def plot_hht(self, fs, T, f, hht, MS):
        """Hilbert谱 和 边际谱 绘图用代码
        输入：
            fs：采样频率
            T：数据时长，注意：锯齿矩阵中的数据时长不一致！
            f：频率坐标
            hht：每个时间点对应频率坐标变换结果
            MS：边际谱
        """
        fig = plt.figure(figsize=(20, 5))
        a = plt.pcolormesh(np.linspace(0, T, T * fs), f, hht, cmap="jet")
        fig.colorbar(a, shrink=0.5)
        fig.colorbar
        plt.title("Hilbert spectrum")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (secs)")
        plt.show()

        fig = plt.figure(figsize=(20, 5))
        plt.plot(f, MS, "b")
        plt.title("Marginal spectrum")
        plt.ylabel("Amplitude")
        plt.xlabel("Frequency")
        plt.show()

    def get_stfs(self, fs, window="hann", nperseg=256, noverlap=None):
        """短时傅里叶
        输入：
            fs：采样频率，
            window：所用窗，带参窗传入tuple，如('gaussian', std)
            nperseg：每窗采样点数
            noverlap：每窗重叠点数
        输出：
            f：频率分辨坐标
            t：每窗时间坐标
            Zxx.real：对应每窗stft变换实值
        """
        if self.mtx_type in (0, 1):
            # 向量
            f, t, Zxx = sig.stft(
                self.data, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap
            )
            return [f.tolist(), t.tolist(), Zxx.real.tolist()]

        if self.mtx_type == 2:
            # 锯齿矩阵
            f_pool, t_pool, Zxx_pool = [], [], []
            for v in self.data:
                f, t, Zxx = sig.stft(
                    np.array(v),
                    fs=fs,
                    window=window,
                    nperseg=nperseg,
                    noverlap=noverlap,
                )
                f_pool.append(list(f))
                t_pool.append(list(t))
                Zxx_pool.append(list(Zxx.real))
            return [f_pool, t_pool, Zxx_pool]

    def get_rnd(self, num=1):
        # 随机数生成，由于引用了np包，直接使用np包随机数生成函数
        return np.random.rand(num).tolist()
