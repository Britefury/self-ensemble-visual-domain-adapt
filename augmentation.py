import numpy as np
import cv2

def identity_xf(N):
    """
    Construct N identity 2x3 transformation matrices
    :return: array of shape (N, 2, 3)
    """
    xf = np.zeros((N, 2, 3), dtype=np.float32)
    xf[:, 0, 0] = xf[:, 1, 1] = 1.0
    return xf

def inv_nx2x2(X):
    """
    Invert the N 2x2 transformation matrices stored in X; a (N,2,2) array
    :param X: transformation matrices to invert, (N,2,2) array
    :return: inverse of X
    """
    rdet = 1.0 / (X[:, 0, 0] * X[:, 1, 1] - X[:, 1, 0] * X[:, 0, 1])
    y = np.zeros_like(X)
    y[:, 0, 0] = X[:, 1, 1] * rdet
    y[:, 1, 1] = X[:, 0, 0] * rdet
    y[:, 0, 1] = -X[:, 0, 1] * rdet
    y[:, 1, 0] = -X[:, 1, 0] * rdet
    return y

def inv_nx2x3(m):
    """
    Invert the N 2x3 transformation matrices stored in X; a (N,2,3) array
    :param X: transformation matrices to invert, (N,2,3) array
    :return: inverse of X
    """
    m2 = m[:, :, :2]
    mx = m[:, :, 2:3]
    m2inv = inv_nx2x2(m2)
    mxinv = np.matmul(m2inv, -mx)
    return np.append(m2inv, mxinv, axis=2)

def cat_nx2x3(a, b):
    """
    Multiply the N 2x3 transformations stored in `a` with those in `b`
    :param a: transformation matrices, (N,2,3) array
    :param b: transformation matrices, (N,2,3) array
    :return: `a . b`
    """
    a2 = a[:, :, :2]
    b2 = b[:, :, :2]

    ax = a[:, :, 2:3]
    bx = b[:, :, 2:3]

    ab2 = np.matmul(a2, b2)
    abx = ax + np.matmul(a2, bx)
    return np.append(ab2, abx, axis=2)

def centre_xf(xf, size):
    """
    Centre the transformations in `xf` around (0,0), where the current centre is assumed to be at the
    centre of an image of shape `size`
    :param xf: transformation matrices, (N,2,3) array
    :param size: image size
    :return: centred transformation matrices, (N,2,3) array
    """
    height, width = size

    # centre_to_zero moves the centre of the image to (0,0)
    centre_to_zero = np.zeros((1, 2, 3), dtype=np.float32)
    centre_to_zero[0, 0, 0] = centre_to_zero[0, 1, 1] = 1.0
    centre_to_zero[0, 0, 2] = -float(width) * 0.5
    centre_to_zero[0, 1, 2] = -float(height) * 0.5

    # centre_to_zero then xf
    xf_centred = cat_nx2x3(xf, centre_to_zero)

    # move (0,0) back to the centre
    xf_centred[:, 0, 2] += float(width) * 0.5
    xf_centred[:, 1, 2] += float(height) * 0.5

    return xf_centred


class ImageAugmentation (object):
    def __init__(self, hflip, xlat_range, affine_std,
                 intens_flip=False,
                 intens_scale_range_lower=None, intens_scale_range_upper=None,
                 intens_offset_range_lower=None, intens_offset_range_upper=None,
                 scale_x_range=None, scale_y_range=None, scale_u_range=None, gaussian_noise_std=0.0,
                 blur_range=None,
                 constrain_hflip=False, constrain_xlat=False, constrain_affine=False,
                 constrain_intens_flip=False, constrain_intens_scale=False, constrain_intens_offset=False,
                 constrain_scale=False):
        self.hflip = hflip
        self.xlat_range = xlat_range
        self.affine_std = affine_std
        self.intens_scale_range_lower = intens_scale_range_lower
        self.intens_scale_range_upper = intens_scale_range_upper
        self.intens_offset_range_lower = intens_offset_range_lower
        self.intens_offset_range_upper = intens_offset_range_upper
        self.intens_flip = intens_flip
        self.scale_x_range = scale_x_range
        self.scale_y_range = scale_y_range
        self.scale_u_range = scale_u_range
        self.gaussian_noise_std = gaussian_noise_std
        self.blur_range = blur_range

        self.constrain_hflip = constrain_hflip
        self.constrain_xlat = constrain_xlat
        self.constrain_affine = constrain_affine
        self.constrain_intens_flip = constrain_intens_flip
        self.constrain_intens_scale = constrain_intens_scale
        self.constrain_intens_offset = constrain_intens_offset
        self.constrain_scale = constrain_scale


    def augment(self, X):
        X = X.copy()
        xf = identity_xf(len(X))

        if self.hflip:
            x_hflip = np.random.binomial(1, 0.5, size=(len(X),)) * 2 - 1
            xf[:, 0, 0] = x_hflip

        if self.scale_x_range is not None and self.scale_x_range[0] is not None:
            xf[:, 0, 0] *= np.random.uniform(low=self.scale_x_range[0], high=self.scale_x_range[1], size=(len(X),))
        if self.scale_y_range is not None and self.scale_y_range[0] is not None:
            xf[:, 1, 1] *= np.random.uniform(low=self.scale_y_range[0], high=self.scale_y_range[1], size=(len(X),))
        if self.scale_u_range is not None and self.scale_u_range[0] is not None:
            scale_u = np.random.uniform(low=self.scale_u_range[0], high=self.scale_u_range[1], size=(len(X),))
            xf[:, 0, 0] *= scale_u
            xf[:, 1, 1] *= scale_u

        if self.affine_std > 0.0:
            xf[:, :, :2] += np.random.normal(scale=self.affine_std, size=(len(X), 2, 2))
        if self.xlat_range > 0.0:
            xf[:, :, 2:] += np.random.uniform(low=-self.xlat_range, high=self.xlat_range, size=(len(X), 2, 1))

        if self.intens_flip:
            col_factor = (np.random.binomial(1, 0.5, size=(len(X), 1, 1, 1)) * 2 - 1).astype(np.float32)
            X = (X * col_factor).astype(np.float32)

        if self.intens_scale_range_lower is not None:
            col_factor = np.random.uniform(low=self.intens_scale_range_lower, high=self.intens_scale_range_upper,
                                           size=(len(X), 1, 1, 1))

            X = (X * col_factor).astype(np.float32)

        if self.intens_offset_range_lower is not None:
            col_offset = np.random.uniform(low=self.intens_offset_range_lower, high=self.intens_offset_range_upper,
                                           size=(len(X), 1, 1, 1))

            X = (X + col_offset).astype(np.float32)

        xf_centred = centre_xf(xf, X.shape[2:])
        for i in range(len(X)):
            X[i, 0, :, :] = cv2.warpAffine(X[i, 0, :, :], xf_centred[i, :, :], (X.shape[3], X.shape[2]))

        if self.blur_range is not None:
            sigmas = np.random.uniform(low=self.blur_range[0], high=self.blur_range[1], size=(len(X),))
            sigmas = np.maximum(sigmas, 0.0)
            for i in range(len(X)):
                sigma = sigmas[i]
                ksize = int(sigma * 8) + 1
                X[i, 0, :, :] = cv2.GaussianBlur(X[i, 0, :, :], (ksize, ksize), sigmaX=sigma)

        if self.gaussian_noise_std > 0.0:
            X += np.random.normal(scale=self.gaussian_noise_std, size=X.shape).astype(np.float32)

        return X


    def augment_pair(self, X):
        X0 = X.copy()
        X1 = X.copy()
        xf0 = identity_xf(len(X))
        xf1 = identity_xf(len(X))

        if self.hflip:
            if self.constrain_hflip:
                x_hflip = np.random.binomial(1, 0.5, size=(len(X),)) * 2 - 1
                xf0[:, 0, 0] = xf1[:, 0, 0] = x_hflip
            else:
                x_hflip0 = np.random.binomial(1, 0.5, size=(len(X),)) * 2 - 1
                x_hflip1 = np.random.binomial(1, 0.5, size=(len(X),)) * 2 - 1
                xf0[:, 0, 0] = x_hflip0
                xf1[:, 0, 0] = x_hflip1

        if self.scale_x_range is not None and self.scale_x_range[0] is not None:
            if self.constrain_scale:
                scale_x = np.random.uniform(low=self.scale_x_range[0], high=self.scale_x_range[1], size=(len(X),))
                xf0[:, 0, 0] *= scale_x
                xf1[:, 0, 0] *= scale_x
            else:
                xf0[:, 0, 0] *= np.random.uniform(low=self.scale_x_range[0], high=self.scale_x_range[1], size=(len(X),))
                xf1[:, 0, 0] *= np.random.uniform(low=self.scale_x_range[0], high=self.scale_x_range[1], size=(len(X),))

        if self.scale_y_range is not None and self.scale_y_range[0] is not None:
            if self.constrain_scale:
                scale_y = np.random.uniform(low=self.scale_y_range[0], high=self.scale_y_range[1], size=(len(X),))
                xf0[:, 1, 1] *= scale_y
                xf1[:, 1, 1] *= scale_y
            else:
                xf0[:, 1, 1] *= np.random.uniform(low=self.scale_y_range[0], high=self.scale_y_range[1], size=(len(X),))
                xf1[:, 1, 1] *= np.random.uniform(low=self.scale_y_range[0], high=self.scale_y_range[1], size=(len(X),))

        if self.scale_u_range is not None and self.scale_u_range[0] is not None:
            if self.constrain_scale:
                scale_u = np.random.uniform(low=self.scale_u_range[0], high=self.scale_u_range[1], size=(len(X),))
                xf0[:, 0, 0] *= scale_u
                xf0[:, 1, 1] *= scale_u
                xf1[:, 0, 0] *= scale_u
                xf1[:, 1, 1] *= scale_u
            else:
                scale_u0 = np.random.uniform(low=self.scale_u_range[0], high=self.scale_u_range[1], size=(len(X),))
                scale_u1 = np.random.uniform(low=self.scale_u_range[0], high=self.scale_u_range[1], size=(len(X),))
                xf0[:, 0, 0] *= scale_u0
                xf0[:, 1, 1] *= scale_u0
                xf1[:, 0, 0] *= scale_u1
                xf1[:, 1, 1] *= scale_u1

        if self.affine_std > 0.0:
            if self.constrain_affine:
                affine = np.random.normal(scale=self.affine_std, size=(len(X), 2, 2))
                xf0[:, :, :2] += affine
                xf1[:, :, :2] += affine
            else:
                xf0[:, :, :2] += np.random.normal(scale=self.affine_std, size=(len(X), 2, 2))
                xf1[:, :, :2] += np.random.normal(scale=self.affine_std, size=(len(X), 2, 2))

        if self.xlat_range > 0.0:
            if self.constrain_xlat:
                xlat = np.random.uniform(low=-self.xlat_range, high=self.xlat_range, size=(len(X), 2, 1))
                xf0[:, :, 2:] += xlat
                xf1[:, :, 2:] += xlat
            else:
                xf0[:, :, 2:] += np.random.uniform(low=-self.xlat_range, high=self.xlat_range, size=(len(X), 2, 1))
                xf1[:, :, 2:] += np.random.uniform(low=-self.xlat_range, high=self.xlat_range, size=(len(X), 2, 1))

        if self.intens_flip:
            if self.constrain_intens_flip:
                col_factor = (np.random.binomial(1, 0.5, size=(len(X), 1, 1, 1)) * 2 - 1).astype(np.float32)
                X0 = (X0 * col_factor).astype(np.float32)
                X1 = (X1 * col_factor).astype(np.float32)
            else:
                col_factor0 = (np.random.binomial(1, 0.5, size=(len(X), 1, 1, 1)) * 2 - 1).astype(np.float32)
                col_factor1 = (np.random.binomial(1, 0.5, size=(len(X), 1, 1, 1)) * 2 - 1).astype(np.float32)
                X0 = (X0 * col_factor0).astype(np.float32)
                X1 = (X1 * col_factor1).astype(np.float32)

        if self.intens_scale_range_lower is not None:
            if self.constrain_intens_scale:
                col_factor = np.random.uniform(low=self.intens_scale_range_lower, high=self.intens_scale_range_upper,
                                               size=(len(X), 1, 1, 1))

                X0 = (X0 * col_factor).astype(np.float32)
                X1 = (X1 * col_factor).astype(np.float32)
            else:
                col_factor0 = np.random.uniform(low=self.intens_scale_range_lower, high=self.intens_scale_range_upper,
                                                size=(len(X), 1, 1, 1))
                col_factor1 = np.random.uniform(low=self.intens_scale_range_lower, high=self.intens_scale_range_upper,
                                                size=(len(X), 1, 1, 1))

                X0 = (X0 * col_factor0).astype(np.float32)
                X1 = (X1 * col_factor1).astype(np.float32)

        if self.intens_offset_range_lower is not None:
            if self.constrain_intens_offset:
                col_offset = np.random.uniform(low=self.intens_offset_range_lower, high=self.intens_offset_range_upper,
                                               size=(len(X), 1, 1, 1))

                X0 = (X0 + col_offset).astype(np.float32)
                X1 = (X1 + col_offset).astype(np.float32)
            else:
                col_offset0 = np.random.uniform(low=self.intens_offset_range_lower, high=self.intens_offset_range_upper,
                                                size=(len(X), 1, 1, 1))
                col_offset1 = np.random.uniform(low=self.intens_offset_range_lower, high=self.intens_offset_range_upper,
                                                size=(len(X), 1, 1, 1))

                X0 = (X0 + col_offset0).astype(np.float32)
                X1 = (X1 + col_offset1).astype(np.float32)

        xf0_centred = centre_xf(xf0, X.shape[2:])
        xf1_centred = centre_xf(xf1, X.shape[2:])
        for i in range(len(X)):
            X0[i, 0, :, :] = cv2.warpAffine(X0[i, 0, :, :], xf0_centred[i, :, :], (X0.shape[3], X0.shape[2]))
            X1[i, 0, :, :] = cv2.warpAffine(X1[i, 0, :, :], xf1_centred[i, :, :], (X1.shape[3], X1.shape[2]))

        if self.gaussian_noise_std > 0.0:
            X0 += np.random.normal(scale=self.gaussian_noise_std, size=X0.shape).astype(np.float32)
            X1 += np.random.normal(scale=self.gaussian_noise_std, size=X1.shape).astype(np.float32)

        return X0, X1
