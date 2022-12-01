import csv
import datetime as dt
from typing import Dict, List, Optional, Tuple, Union

import gpflow
import numpy as np
import pandas as pd
import tensorflow as tf
from gpflow.kernels import ChangePoints, Matern12, Matern32, Matern52, Stationary
from sklearn.preprocessing import StandardScaler
from tensorflow_probability import bijectors as tfb

import sys
sys.path.append("/scratch/yk2516/currency-change-point-detection")
from mom_trans.spectralmixture import SpectralMixture, sm_init

Kernel = gpflow.kernels.base.Kernel

MAX_ITERATIONS = 200

class ChangePointsWithBounds(ChangePoints):
    def __init__(
        self,
        kernels: Tuple[Kernel, Kernel],
        location: float,
        interval: Tuple[float, float],
        steepness: float = 1.0,
        name: Optional[str] = None,
    ):
        """Overwrite the Chnagepoints class to
        1) only take a single location
        2) so location is bounded by interval

        Args:
            kernels (Tuple[Kernel, Kernel]): the left hand and right hand kernels
            location (float): changepoint location initialisation, must lie within interval
            interval (Tuple[float, float]): the interval which bounds the changepoint hyperparameter
            steepness (float, optional): initialisation of the steepness parameter. Defaults to 1.0.
            name (Optional[str], optional): class name. Defaults to None.

        Raises:
            ValueError: errors if intial changepoint location is not within interval
        """
        # overwrite the locations variable to enforce bounds
        if location < interval[0] or location > interval[1]:
            raise ValueError(
                "Location {loc} is not in range [{low},{high}]".format(
                    loc=location, low=interval[0], high=interval[1]
                )
            )
        locations = [location]
        super().__init__(
            kernels=kernels, locations=locations, steepness=steepness, name=name
        )

        affine = tfb.Shift(tf.cast(interval[0], tf.float64))(
            tfb.Scale(tf.cast(interval[1] - interval[0], tf.float64))
        )
        self.locations = gpflow.base.Parameter(
            locations, transform=tfb.Chain([affine, tfb.Sigmoid()]), dtype=tf.float64
        )

    def _sigmoids(self, X: tf.Tensor) -> tf.Tensor:
        # overwrite to remove sorting of locations
        locations = tf.reshape(self.locations, (1, 1, -1))
        steepness = tf.reshape(self.steepness, (1, 1, -1))
        return tf.sigmoid(steepness * (X[:, :, None] - locations))

def fit_matern_kernel(
    time_series_data: pd.DataFrame,
    variance: float = 1.0,
    lengthscale: float = 1.0,
    likelihood_variance: float = 1.0,
    kernel_choice="Matern32",
    num_mixtures=5
) -> Tuple[float, Dict[str, float]]:
    """Fit the Matern 3/2 kernel on a time-series

    Args:
        time_series_data (pd.DataFrame): time-series with columns X and Y
        variance (float, optional): variance parameter initialisation. Defaults to 1.0.
        lengthscale (float, optional): lengthscale parameter initialisation. Defaults to 1.0.
        likelihood_variance (float, optional): likelihood variance parameter initialisation. Defaults to 1.0.

    Returns:
        Tuple[float, Dict[str, float]]: negative log marginal likelihood and paramters after fitting the GP
    """
    current_kernel = None

    if kernel_choice == "Matern12":
        current_kernel = Matern12(variance=variance, lengthscales=lengthscale)
    elif kernel_choice == "Matern32":
        current_kernel = Matern32(variance=variance, lengthscales=lengthscale)
    elif kernel_choice == "Matern52":
        current_kernel = Matern52(variance=variance, lengthscales=lengthscale)
    elif kernel_choice == "SpectralMixture":
        # print(f"time_series_data.loc[:, ['X']].to_numpy().shape = {time_series_data.loc[:, ['X']].to_numpy().shape}")
        D = time_series_data.loc[:, ['X']].to_numpy().shape[1] # number of dimensions

        weights, means, scales = sm_init(
            train_x = time_series_data.loc[:, ["X"]].to_numpy(), 
            train_y = time_series_data.loc[:, ["Y"]].to_numpy(),
            num_mixtures = num_mixtures
        )
        # print(f"weights = {weights}")
        # print(f"means = {means}")
        # print(f"scales = {scales}")

        if 0 in means:
            means[np.where(means==0)[0].item()] += 0.01
            print(f"means (updated) = {means}")

        current_kernel = SpectralMixture(num_mixtures=num_mixtures, mixture_weights=weights, mixture_scales=scales, mixture_means=means, input_dim=D)
    elif kernel_choice == "Matern12_32":
        current_kernel = Matern12(variance=variance, lengthscales=lengthscale) + Matern32(variance=variance, lengthscales=lengthscale)
    elif kernel_choice == "Matern12_52":
        current_kernel = Matern12(variance=variance, lengthscales=lengthscale) + Matern52(variance=variance, lengthscales=lengthscale)
    else:
        raise NotImplementedError

    print(f"+++++ current_kernel = {current_kernel} +++++")
    if kernel_choice == "SpectralMixture":
        print(f"+++++ num_mixtures = {num_mixtures} +++++")

    m = gpflow.models.GPR(
        data=(
            time_series_data.loc[:, ["X"]].to_numpy(),
            time_series_data.loc[:, ["Y"]].to_numpy(),
        ),
        kernel=current_kernel,
        noise_variance=likelihood_variance,
    )
    opt = gpflow.optimizers.Scipy()
    nlml = opt.minimize(
        m.training_loss, m.trainable_variables, options=dict(maxiter=MAX_ITERATIONS)
    ).fun

    print(f"m.trainable_variables={m.trainable_variables}")

    # try:
    #     # these lines works! 
    #     # m.kernel.kernels[0]=<gpflow.kernels.stationaries.Matern12 object at 0x1473fa05ea30>
    #     # m.kernel.kernels[0].variance.numpy()=2.162948539142556e-06
    #     # m.kernel.kernels[1]=<gpflow.kernels.stationaries.Matern32 object at 0x1473fa05ec40>
    #     # m.kernel.kernels[1].variance.numpy()=1.001339392972587
        
    #     print(f"m.kernel.kernels[0]={m.kernel.kernels[0]}")
    #     print(f"m.kernel.kernels[0].variance.numpy()={m.kernel.kernels[0].variance.numpy()}")
    #     print(f"m.kernel.kernels[1]={m.kernel.kernels[1]}")
    #     print(f"m.kernel.kernels[1].variance.numpy()={m.kernel.kernels[1].variance.numpy()}")
    # except:
    #     print(f"m.kernel[0] = {m.kernel[0]}")
    #     print(f"m.kernel[0].variance.numpy() = {m.kernel[0].variance.numpy()}")
    #     print(f"m.kernel[0] = {m.kernel[1]}")

    if kernel_choice == "SpectralMixture":
        params = {
            "kM_num_mixtures": m.kernel.num_mixtures.numpy(),
            "kM_mixture_weights": m.kernel.mixture_weights.numpy(),
            "kM_mixture_scales": m.kernel.mixture_scales.numpy(),
            "kM_mixture_means": m.kernel.mixture_means.numpy(),
            "kM_input_dim": m.kernel.input_dim,
            "kM_active_dims": m.kernel.active_dims,
            "kM_likelihood_variance": m.likelihood.variance.numpy(),
        }
    elif "_" in kernel_choice: # sums of kernels
        params = {
            "kM_variance": [m.kernel.kernels[0].variance.numpy(), m.kernel.kernels[1].variance.numpy()],
            "kM_lengthscales": [m.kernel.kernels[0].lengthscales.numpy(),m.kernel.kernels[1].lengthscales.numpy()],
            "kM_likelihood_variance": [m.likelihood.variance.numpy()],
        }
    else:
        params = {
            "kM_variance": m.kernel.variance.numpy(),
            "kM_lengthscales": m.kernel.lengthscales.numpy(),
            "kM_likelihood_variance": m.likelihood.variance.numpy(),
        }

    return nlml, params

def fit_changepoint_kernel(
    time_series_data: pd.DataFrame,
    k1_variance: float = 1.0,
    k1_lengthscale: float = 1.0,
    k2_variance: float = 1.0,
    k2_lengthscale: float = 1.0,
    kC_likelihood_variance=1.0,
    kC_changepoint_location=None,
    kC_steepness=1.0,
    kernel_choice="Matern32",

    # SM Kernel
    k1_num_mixtures = None,
    k1_mixture_weights = None,
    k1_mixture_scales = None,
    k1_mixture_means = None,
    k1_input_dim = None,
    k1_active_dims = None,

    k2_num_mixtures = None,
    k2_mixture_weights = None,
    k2_mixture_scales = None,
    k2_mixture_means = None,
    k2_input_dim = None,
    k2_active_dims = None,
) -> Tuple[float, float, Dict[str, float]]:
    """Fit the Changepoint kernel on a time-series

    Args:
        time_series_data (pd.DataFrame): time-series with ciolumns X and Y
        k1_variance (float, optional): variance parameter initialisation for k1. Defaults to 1.0.
        k1_lengthscale (float, optional): lengthscale initialisation for k1. Defaults to 1.0.
        k2_variance (float, optional): variance parameter initialisation for k2. Defaults to 1.0.
        k2_lengthscale (float, optional): lengthscale initialisation for k2. Defaults to 1.0.
        kC_likelihood_variance (float, optional): likelihood variance parameter initialisation. Defaults to 1.0.
        kC_changepoint_location (float, optional): changepoint location initialisation, if None uses midpoint of interval. Defaults to None.
        kC_steepness (float, optional): steepness parameter initialisation. Defaults to 1.0.

    Returns:
        Tuple[float, float, Dict[str, float]]: changepoint location, negative log marginal likelihood and paramters after fitting the GP
    """
    kernel1 = None
    kernel2 = None

    if kernel_choice == "Matern12":
        kernel1 = Matern12(variance=k1_variance, lengthscales=k1_lengthscale)
        kernel2 = Matern12(variance=k2_variance, lengthscales=k2_lengthscale)
    elif kernel_choice == "Matern32":
        kernel1 = Matern32(variance=k1_variance, lengthscales=k1_lengthscale)
        kernel2 = Matern32(variance=k2_variance, lengthscales=k2_lengthscale)
    elif kernel_choice == "Matern52":
        kernel1 = Matern52(variance=k1_variance, lengthscales=k1_lengthscale)
        kernel2 = Matern52(variance=k2_variance, lengthscales=k2_lengthscale)
    elif kernel_choice == "SpectralMixture":
        kernel1 = SpectralMixture(num_mixtures=k1_num_mixtures, mixture_weights=k1_mixture_weights, mixture_scales=k1_mixture_scales, mixture_means=k1_mixture_means, input_dim=k1_input_dim, active_dims=k1_active_dims)
        kernel2 = SpectralMixture(num_mixtures=k2_num_mixtures, mixture_weights=k2_mixture_weights, mixture_scales=k2_mixture_scales, mixture_means=k2_mixture_means, input_dim=k2_input_dim, active_dims=k2_active_dims)
    elif kernel_choice == "Matern12_32":
        kernel1 = Matern12(variance=k1_variance[0], lengthscales=k1_lengthscale[0]) + Matern32(variance=k1_variance[1], lengthscales=k1_lengthscale[1])
        kernel2 = Matern12(variance=k2_variance[0], lengthscales=k2_lengthscale[0]) + Matern32(variance=k2_variance[1], lengthscales=k2_lengthscale[1])
    elif kernel_choice == "Matern12_52":
        kernel1 = Matern12(variance=k1_variance[0], lengthscales=k1_lengthscale[0]) + Matern52(variance=k1_variance[1], lengthscales=k1_lengthscale[1])
        kernel2 = Matern12(variance=k2_variance[0], lengthscales=k2_lengthscale[0]) + Matern52(variance=k2_variance[1], lengthscales=k2_lengthscale[1])
    else:
        raise NotImplementedError
        
    print(f"+++++ kernel1 = {kernel1} +++++")
    print(f"+++++ kernel2 = {kernel2} +++++")

    if not kC_changepoint_location:
        kC_changepoint_location = (
            time_series_data["X"].iloc[0] + time_series_data["X"].iloc[-1]
        ) / 2.0

    m = gpflow.models.GPR(
        data=(
            time_series_data.loc[:, ["X"]].to_numpy(),
            time_series_data.loc[:, ["Y"]].to_numpy(),
        ),
        kernel=ChangePointsWithBounds(
            [
                kernel1,
                kernel2,
            ],
            location=kC_changepoint_location,
            interval=(time_series_data["X"].iloc[0], time_series_data["X"].iloc[-1]),
            steepness=kC_steepness,
        ),
    )
    m.likelihood.variance.assign(kC_likelihood_variance)
    opt = gpflow.optimizers.Scipy()
    nlml = opt.minimize(
        m.training_loss, m.trainable_variables, options=dict(maxiter=200)
    ).fun
    changepoint_location = m.kernel.locations[0].numpy()

    if kernel_choice == "SpectralMixture":
        # TODO change this
        params = {}
    elif "_" in kernel_choice: # sums of kernels
        # TODO change this
        params = {}
    else:
        params = {
            "k1_variance": m.kernel.kernels[0].variance.numpy().flatten()[0],
            "k1_lengthscale": m.kernel.kernels[0].lengthscales.numpy().flatten()[0],
            "k2_variance": m.kernel.kernels[1].variance.numpy().flatten()[0],
            "k2_lengthscale": m.kernel.kernels[1].lengthscales.numpy().flatten()[0],
            "kC_likelihood_variance": m.likelihood.variance.numpy().flatten()[0],
            "kC_changepoint_location": changepoint_location,
            "kC_steepness": m.kernel.steepness.numpy(),
        }
    return changepoint_location, nlml, params


def changepoint_severity(
    kC_nlml: Union[float, List[float]], kM_nlml: Union[float, List[float]]
) -> float:
    """Changepoint score as detailed in https://arxiv.org/pdf/2105.13727.pdf

    Args:
        kC_nlml (Union[float, List[float]]): negative log marginal likelihood of Changepoint kernel
        kM_nlml (Union[float, List[float]]): negative log marginal likelihood of Matern 3/2 kernel

    Returns:
        float: changepoint score
    """
    normalized_nlml = kC_nlml - kM_nlml
    return 1 - 1 / (np.mean(np.exp(-normalized_nlml)) + 1)


def changepoint_loc_and_score(
    time_series_data_window: pd.DataFrame,
    kM_variance: float = 1.0,
    kM_lengthscale: float = 1.0,
    kM_likelihood_variance: float = 1.0,
    k1_variance: float = None,
    k1_lengthscale: float = None,
    k2_variance: float = None,
    k2_lengthscale: float = None,
    kC_likelihood_variance=1.0, #TODO note this seems to work better by resetting this
    # kC_likelihood_variance=None,
    kC_changepoint_location=None,
    kC_steepness=1.0,
    kernel_choice="Matern32",
    num_mixtures=5
) -> Tuple[float, float, float, Dict[str, float], Dict[str, float]]:
    """For a single time-series window, calcualte changepoint score and location as detailed in https://arxiv.org/pdf/2105.13727.pdf

    Args:
        time_series_data_window (pd.DataFrame): time-series with columns X and Y
        kM_variance (float, optional): variance initialisation for Matern 3/2 kernel. Defaults to 1.0.
        kM_lengthscale (float, optional): lengthscale initialisation for Matern 3/2 kernel. Defaults to 1.0.
        kM_likelihood_variance (float, optional): likelihood variance initialisation for Matern 3/2 kernel. Defaults to 1.0.
        k1_variance (float, optional): variance initialisation for Changepoint kernel k1, if None uses fitted variance parameter from Matern 3/2. Defaults to None.
        k1_lengthscale (float, optional): lengthscale initialisation for Changepoint kernel k1, if None uses fitted lengthscale parameter from Matern 3/2. Defaults to None.
        k2_variance (float, optional): variance initialisation for Changepoint kernel k2, if None uses fitted variance parameter from Matern 3/2. Defaults to None.
        k2_lengthscale (float, optional): lengthscale initialisation for for Changepoint kernel k2, if None uses fitted lengthscale parameter from Matern 3/2. Defaults to None.
        kC_likelihood_variance ([type], optional): likelihood variance initialisation for Changepoint kernel. Defaults to None.
        kC_changepoint_location ([type], optional): changepoint location initialisation for Changepoint, if None uses midpoint of interval. Defaults to None.
        kC_steepness (float, optional): changepoint location initialisation for Changepoint. Defaults to 1.0.

    Returns:
        Tuple[float, float, float, Dict[str, float], Dict[str, float]]: changepoint score, changepoint location,
        changepoint location normalised by interval length to [0,1], Matern 3/2 kernel parameters, Changepoint kernel parameters
    """

    time_series_data = time_series_data_window.copy()
    Y_data = time_series_data[["Y"]].values
    time_series_data[["Y"]] = StandardScaler().fit(Y_data).transform(Y_data)
    # time_series_data.loc[:, "X"] = time_series_data.loc[:, "X"] - time_series_data.loc[time_series_data.index[0], "X"]

    try:
        (kM_nlml, kM_params) = fit_matern_kernel(
            time_series_data, kM_variance, kM_lengthscale, kM_likelihood_variance,kernel_choice=kernel_choice, num_mixtures=num_mixtures
        )
    except BaseException as ex:
        # do not want to optimise again if the hyperparameters
        # were already initialised as the defaults
        if kM_variance == kM_lengthscale == kM_likelihood_variance == 1.0:
            raise BaseException(
                "Retry with default hyperparameters - already using default parameters."
            ) from ex
        (
            kM_nlml,
            kM_params,
        ) = fit_matern_kernel(time_series_data,kernel_choice=kernel_choice)
    
    is_cp_location_default = (
        (not kC_changepoint_location)
        or kC_changepoint_location < time_series_data["X"].iloc[0]
        or kC_changepoint_location > time_series_data["X"].iloc[-1]
    )
    if is_cp_location_default:
        # default to midpoint
        kC_changepoint_location = (
            time_series_data["X"].iloc[-1] + time_series_data["X"].iloc[0]
        ) / 2.0

    print(f"kM_params={kM_params}")

    if kernel_choice == "SpectralMixture":
        k1_variance = None
        k1_lengthscale = None
        k2_variance = None
        k2_lengthscale = None
        kC_likelihood_variance = kM_params["kM_likelihood_variance"]

        k1_num_mixtures = kM_params["kM_num_mixtures"]
        k1_mixture_weights = kM_params["kM_mixture_weights"]
        k1_mixture_scales = kM_params["kM_mixture_scales"]
        k1_mixture_means = kM_params["kM_mixture_means"]
        k1_input_dim = kM_params["kM_input_dim"]
        k1_active_dims = kM_params["kM_active_dims"]

        k2_num_mixtures = kM_params["kM_num_mixtures"]
        k2_mixture_weights = kM_params["kM_mixture_weights"]
        k2_mixture_scales = kM_params["kM_mixture_scales"]
        k2_mixture_means = kM_params["kM_mixture_means"]
        k2_input_dim = kM_params["kM_input_dim"]
        k2_active_dims = kM_params["kM_active_dims"]
    elif "_" in kernel_choice: # sums of kernels
        if not k1_variance:
            k1_variance = [kM_params["kM_variance"][0],kM_params["kM_variance"][1]]

        if not k1_lengthscale:
            k1_lengthscale = [kM_params["kM_lengthscales"][0], kM_params["kM_lengthscales"][1]]

        if not k2_variance:
            k2_variance = [kM_params["kM_variance"][0], kM_params["kM_variance"][1]]

        if not k2_lengthscale:
            k2_lengthscale = [kM_params["kM_lengthscales"][0], kM_params["kM_lengthscales"][1]]

        if not kC_likelihood_variance:
            kC_likelihood_variance = kM_params["kM_likelihood_variance"][0]
        
        k1_num_mixtures = None
        k1_mixture_weights = None
        k1_mixture_scales = None
        k1_mixture_means = None
        k1_input_dim = None
        k1_active_dims = None

        k2_num_mixtures = None
        k2_mixture_weights = None
        k2_mixture_scales = None
        k2_mixture_means = None
        k2_input_dim = None
        k2_active_dims = None

    else:
        if not k1_variance:
            k1_variance = kM_params["kM_variance"]

        if not k1_lengthscale:
            k1_lengthscale = kM_params["kM_lengthscales"]

        if not k2_variance:
            k2_variance = kM_params["kM_variance"]

        if not k2_lengthscale:
            k2_lengthscale = kM_params["kM_lengthscales"]

        if not kC_likelihood_variance:
            kC_likelihood_variance = kM_params["kM_likelihood_variance"]

        k1_num_mixtures = None
        k1_mixture_weights = None
        k1_mixture_scales = None
        k1_mixture_means = None
        k1_input_dim = None
        k1_active_dims = None

        k2_num_mixtures = None
        k2_mixture_weights = None
        k2_mixture_scales = None
        k2_mixture_means = None
        k2_input_dim = None
        k2_active_dims = None

    try:
        (changepoint_location, kC_nlml, kC_params) = fit_changepoint_kernel(
            time_series_data,
            k1_variance=k1_variance,
            k1_lengthscale=k1_lengthscale,
            k2_variance=k2_variance,
            k2_lengthscale=k2_lengthscale,
            kC_likelihood_variance=kC_likelihood_variance,
            kC_changepoint_location=kC_changepoint_location,
            kC_steepness=kC_steepness,
            kernel_choice=kernel_choice,

            # Spectral Mixture Kernel parameters
            k1_num_mixtures = k1_num_mixtures,
            k1_mixture_weights = k1_mixture_weights,
            k1_mixture_scales = k1_mixture_scales,
            k1_mixture_means = k1_mixture_means,
            k1_input_dim = k1_input_dim,
            k1_active_dims = k1_active_dims,

            k2_num_mixtures = k2_num_mixtures,
            k2_mixture_weights = k2_mixture_weights,
            k2_mixture_scales = k2_mixture_scales,
            k2_mixture_means = k2_mixture_means,
            k2_input_dim = k2_input_dim,
            k2_active_dims = k2_active_dims,

        )
    except BaseException as ex:
        # do not want to optimise again if the hyperparameters
        # were already initialised as the defaults
        if (
            k1_variance
            == k1_lengthscale
            == k2_variance
            == k2_lengthscale
            == kC_likelihood_variance
            == kC_steepness
            == 1.0
        ) and is_cp_location_default:
            raise BaseException(
                "Retry with default hyperparameters - already using default parameters."
            ) from ex
        (
            changepoint_location,
            kC_nlml,
            kC_params,
        ) = fit_changepoint_kernel(time_series_data, kernel_choice=kernel_choice)

    cp_score = changepoint_severity(kC_nlml, kM_nlml)
    cp_loc_normalised = (time_series_data["X"].iloc[-1] - changepoint_location) / (
        time_series_data["X"].iloc[-1] - time_series_data["X"].iloc[0]
    )

    return cp_score, changepoint_location, cp_loc_normalised, kM_params, kC_params


def run_module(
    time_series_data: pd.DataFrame,
    lookback_window_length: int,
    output_csv_file_path: str,
    start_date: dt.datetime = None,
    end_date: dt.datetime = None,
    kernel_choice: str = "Matern32",
    use_kM_hyp_to_initialise_kC=True,

    # Spectral Mixture Parameter
    num_mixtures = 5
):
    """Run the changepoint detection module as described in https://arxiv.org/pdf/2105.13727.pdf
    for all times (in date range if specified). Outputs results to a csv.

    Args:
        time_series_data (pd.DataFrame): time series with date as index and with column daily_returns
        lookback_window_length (int): lookback window length
        output_csv_file_path (str): dull path, including csv extension to output results
        start_date (dt.datetime, optional): start date for module, if None use all (with burnin in period qualt to length of LBW). Defaults to None.
        end_date (dt.datetime, optional): end date for module. Defaults to None.
        use_kM_hyp_to_initialise_kC (bool, optional): initialise Changepoint kernel parameters using the paremters from fitting Matern 3/2 kernel. Defaults to True.
    """
    if start_date and end_date:
        first_window = time_series_data.loc[:start_date].iloc[
            -(lookback_window_length + 1) :, :
        ]
        remaining_data = time_series_data.loc[start_date:end_date, :]
        if remaining_data.index[0] == start_date:
            remaining_data = remaining_data.iloc[1:, :]
        else:
            first_window = first_window.iloc[1:]
        time_series_data = pd.concat([first_window, remaining_data]).copy()
    elif not start_date and not end_date:
        time_series_data = time_series_data.copy()
    elif not start_date:
        time_series_data = time_series_data.iloc[:end_date, :].copy()
    elif not end_date:
        first_window = time_series_data.loc[:start_date].iloc[
            -(lookback_window_length + 1) :, :
        ]
        remaining_data = time_series_data.loc[start_date:, :]
        if remaining_data.index[0] == start_date:
            remaining_data = remaining_data.iloc[1:, :]
        else:
            first_window = first_window.iloc[1:]
        time_series_data = pd.concat([first_window, remaining_data]).copy()

    csv_fields = ["date", "t", "cp_location", "cp_location_norm", "cp_score"]
    with open(output_csv_file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(csv_fields)

    time_series_data["date"] = time_series_data.index
    time_series_data = time_series_data.reset_index(drop=True)
    for window_end in range(lookback_window_length + 1, len(time_series_data)):
        ts_data_window = time_series_data.iloc[
            window_end - (lookback_window_length + 1) : window_end
        ][["date", "daily_returns"]].copy()
        ts_data_window["X"] = ts_data_window.index.astype(float)
        ts_data_window = ts_data_window.rename(columns={"daily_returns": "Y"})
        time_index = window_end - 1
        window_date = ts_data_window["date"].iloc[-1].strftime("%Y-%m-%d")

        try:
            if use_kM_hyp_to_initialise_kC:
                cp_score, cp_loc, cp_loc_normalised, _, _ = changepoint_loc_and_score(
                    ts_data_window,kernel_choice=kernel_choice,num_mixtures=num_mixtures
                )
            else:
                cp_score, cp_loc, cp_loc_normalised, _, _ = changepoint_loc_and_score(
                    ts_data_window,
                    k1_lengthscale=1.0,
                    k1_variance=1.0,
                    k2_lengthscale=1.0,
                    k2_variance=1.0,
                    kC_likelihood_variance=1.0,
                    kernel_choice=kernel_choice,
                    num_mixtures=num_mixtures
                )

        except Exception as e:
            print(e)
            # write as NA when fails and will deal with this later
            cp_score, cp_loc, cp_loc_normalised = "NA", "NA", "NA"

        # #write the reults to the csv
        with open(output_csv_file_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [window_date, time_index, cp_loc, cp_loc_normalised, cp_score]
            )
