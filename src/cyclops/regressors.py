"""
Regression classes for cyclops.

Handles the generation of predicted fields from sensor data.

(c) Copyright UKAEA 2023.
"""
import numpy as np
import holoviews as hv
hv.extension('matplotlib')
from collections import deque
from scipy.interpolate import (
    RBFInterpolator,
    CloughTocher2DInterpolator,
    CubicSpline,
    LinearNDInterpolator,
    RegularGridInterpolator,
    griddata
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")


class RegressionModel:
    """Regression model base class.

    Three core methods:
    1. Initialisation with correct hyperparameters.
    2. Fitting with training data.
    3. Predicting values.

    They all predict 1D outputs only.
    Some models can take in a variety of different input dimensions.
    """

    def __init__(self, num_input_dim: int, min_length: int) -> None:
        """Initialise class instance.

        All regression models use a scaler to rescale the input data and have
        a regressor. The number of dimensions is specified to mitigate errors.

        Args:
            num_input_dim (int): number of dimensions/features of training (and
                test) data.
            min_length (int): minimum length of training dataset.
        """
        self._scaler = preprocessing.StandardScaler()
        self._regressor = None
        self._x_dim = num_input_dim
        self._min_length = min_length

    def prepare_fit(
        self, *train_args: np.ndarray[float]) -> (np.ndarray[float], list[tuple]):
        """Check training data dimensions #and normalise#.

        Args:
            train_x (np.ndarray[float]): n by d array of n input data values of
                dimension d.
            train_y (np.ndarray[float]): n by 1 array of n output data values.

        Returns:
            np.ndarray[float]: scaled n by d array of n input data values of
                dimension d.
        """
        
        
        
        reshaped_train_args = []

        for dim_array in train_args[0][1:]:
            #print("heres the size", dim_array.size)
    
            reshaped_train_args.append(dim_array.reshape(dim_array.shape))

            
        for each_array, i in zip(train_args[0], range(len(train_args[0]))):
            #print(each_array)
            train_args[0][i] = each_array.flatten()
            #print(each_array.flatten())
            #print(train_args[0][i].shape)
        
        

        scaler = self._scaler.fit(train_args[0])
        scaled_output = scaler.transform(train_args[0])
        #scale2 = self._scaler.fit(train_args[0][2:-1])
        #scaled_output2 = self._scaler.transform(train_args[0][2:-1])
        
        all_scaled_output = scaled_output#1 + scaled_output2
        #print(all_scaled_output)
        return all_scaled_output, og_shapes

    def prepare_predict(
        self, predict_x: np.ndarray[float]
    ) -> np.ndarray[float]:
        """Check prediction data dimensions and normalise.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input data values
                of dimension d.

        Returns:
            np.ndarray[float]: scaled n by d array of n input data values of
                dimension d.
        """
        self.check_dim(len(predict_x[0]), self._x_dim, "Input")
        self.check_dim(len(predict_x[0]), self._x_dim, "predict")
        return self._scaler.transform(predict_x)

    def check_dim(self, dim: int, correct_dim: int, data_name: str) -> None:
        """Check the dimensions/features are equal to a specified number.

        Args:
            dim (int): measured number of dimensions.
            correct_dim (int): expected number of dimensions.
            data_name (str): name for exception handling.

        Raises:
            Exception: error to explain user's mistake.
        """
#        if dim != correct_dim:
#            raise Exception(
#                data_name
#                + " data should be a numpy array of shape (-1, "
#                + str(correct_dim)
#                + ")."
#            )

    def check_length(self, length: int) -> None:
        """Check the number of training data is above a minimum length.

        Args:
            length (int): number of training data points.

        Raises:
            Exception: error to explain user's mistake.
        """
        if length < self._min_length:
            raise Exception(
                f"Input data should have a length of >= {self._min_length}."
            )


class RBFModel(RegressionModel):
    """Radial basis function regressor.

    Uses RBF interpolation. Interpolates and extrapolates. Acts in any
    dimension d >= 1. Learns from any number of training data points n >= 2.
    Time complexity of around O(n^3).
    """

    def __init__(self, num_input_dim: int) -> None:
        """Initialise class instance.

        Args:
            num_input_dim (int): number of features (dimensions) for the
                training data.

        Raises:
            Exception: error to explain user's mistake.
        """
        self._min_length = min_length
        self._min_length = min_length
        self._min_length = min_length
        super().__init__(num_input_dim, 2)
        if num_input_dim <= 0:
            raise Exception("Input data should have d >= 1 dimensions.")

    def fit(
        self, train_x: np.ndarray[float], train_y: np.ndarray[float]
    ) -> None:
        """Fit the model to some training data.

        Args:
            train_x (np.ndarray[float]): n by d array of n training inputs with
                d dimensions.
            train_y (np.ndarray[float]): n by 1 array of n training outputs.
        """
        scaled_x = self.prepare_fit(train_x, train_y)
        self._regressor = RegularGridInterpolator((x, y, z), field_data, method='linear')

    def predict(self, predict_x: np.ndarray[float]) -> np.ndarray[float]:
        """Return n predicted outputs of dimension 1 given inputs.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of
                d dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n predicted 1D values.
        """
        scaled_x = self.prepare_predict(predict_x)
        return self._regressor(scaled_x).reshape(-1, 1)


class LModel(RegressionModel):
    """Linear regressor.

    Uses linear splines. Only interpolates. Acts in any dimension d > 1. Learns
    from any number of training data points n >= 3. Time complexity of around
    O(n).
    """

    def __init__(self, num_input_dim) -> None:
        """Initialise class instance.

        Args:
            num_input_dim (int): number of features (dimensions) for the
                training data.

        Raises:
            Exception: error to explain user's mistake.
        """
        super().__init__(num_input_dim, 3)
        if num_input_dim <= 1:
            raise Exception("Input data should have d >= 2 dimensions.")

    def fit(
        self, train_x: np.ndarray[float], train_y: np.ndarray[float]
    ) -> None:
        """Fit the model to some training data.

        Args:
            train_x (np.ndarray[float]): n by d array of n training inputs with
                d dimensions.
            train_y (np.ndarray[float]): n by 1 array of n training outputs.
        """
        scaled_x = self.prepare_fit(train_x, train_y)
        self._regressor = np.polyfit(scaled_x[:,0], train_y, deg=self._degree)

    def predict(self, predict_x: np.ndarray[float]) -> np.ndarray[float]:
        """Return n predicted outputs of dimension 1 given inputs.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of d
                dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n predicted 1D values.
        """
        scaled_x = self.prepare_predict(predict_x)
        value = self._regressor(scaled_x).reshape(-1, 1)
        return value

#########################
class RegGridInterp(RegressionModel):
    """Multidimensional interpolation on regular or rectilinear grids.

    The data must be defined on a rectilinear grid; that is, a rectangular grid with even or uneven spacing. Linear, nearest-neighbor, spline interpolations are supported. 
    - Must update description with equivalent data to the following for it: Only interpolates. Acts in any dimension d > 1. Learns
    from any number of training data points n >= 3. Time complexity of around
    O(n).
    """

    def __init__(self, num_input_dim) -> None:
        """Initialise class instance.

        Args:
            num_input_dim (int): number of features (dimensions) for the
                training data.

        Raises:
            Exception: error to explain user's mistake.
        """
        super().__init__(num_input_dim, 3) #come back to generalise this
        #if num_input_dim <= 1:
        #    raise Exception("Input data should have d >= 2 dimensions.")

    def fit(
        self, pos_data: np.ndarray[float], field_data: np.ndarray[float]) -> None:
        """Fit the model to some training data.

        Args:
            *args (np.ndarray[float]): a series of n by d arrays containing n
            training inputs with d dimensions each. The last argument MUST be values at points described by other arguments.
        """
        
        pos_data = list(pos_data)
         
        x = np.array(pos_data[0])
        y = np.array(pos_data[1])
        z = np.array(pos_data[2])
        
        data = [field_data, x, y, z]
        # We scale the data in order to avoid numerical errors when scales of different dimensions/points are very different
        scaled_data, og_shapes = self.prepare_fit(data)
        
        scaleT = np.array(scaled_data[0])
        scaleX = np.array(scaled_data[1])
        scaleY = np.array(scaled_data[2])
        scaleZ = np.array(scaled_data[3])
        
        xi,yi,zi=np.ogrid[0:1:11j, 0:1:11j, 0:1:11j]
        X1=xi.reshape(xi.shape[0],)
        Y1=yi.reshape(yi.shape[1],)
        Z1=zi.reshape(zi.shape[2],)
        
        ar_len=len(X1)*len(Y1)*len(Z1)
        
        X=np.arange(ar_len,dtype=float)
        Y=np.arange(ar_len,dtype=float)
        Z=np.arange(ar_len,dtype=float)
        
        l=0
        for i in range(0,len(X1)):
            for j in range(0,len(Y1)):
                for k in range(0,len(Z1)):
                    X[l]=X1[i]
                    Y[l]=Y1[j]
                    Z[l]=Z1[k]
                    l=l+1

        #interpolate scaled data on new grid "scaleX,scaleY,scaleZ"
        print("Interpolate...")
        V = griddata((scaleX,scaleY,scaleZ), scaleT, (scaleX,scaleY,scaleZ), method='linear')
        print("griddata completed running...")
        xlim = (min(x), max(x))
        ylim = (min(y), max(y))
        xx = np.arange(min(x), max(x), 1)
        yy = np.arange(min(y), max(y), 1)
        
        zz = griddata((x,y), z, (xx, yy), method="linear")
        print("zz", zz)
        mesh = hv.QuadMesh((xx, yy, zz))
        #img_stk = hv.ImageStack(z, bounds=(min(x), min(y), max(x), max(y)))
        #img_stk
        contour = hv.operation.contours(mesh, levels=8)
        scatter = hv.Scatter((x, y))
        contour_mesh = mesh * contour * scatter
        contour_mesh.redim(
            x=hv.Dimension("x", range=xlim), y=holoviews.Dimension("y", range=ylim),
        ) 

        #xi,yi,zi=np.ogrid[0:1:11j, 0:1:11j, 0:1:11j]
        #print("xi", xi)
        #print("   ")
        #X1=xi.reshape(xi.shape[0],)
        #Y1=yi.reshape(yi.shape[1],)
        #Z1=zi.reshape(zi.shape[2],)
        
        #X1 = x.flatten()
        #print("X1.shape", X1.shape)
        #Y1 = y.flatten()
        #print("Y1.shape", Y1.shape)
        #Z1 = z.flatten()
        

        #V = griddata((x,y,z), v, (X,Y,Z), method='linear')
        
        #return(V, (x, y, z), (X,Y,Z))

 #       grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
 #       grid_temperature = temperatures[0]

        # Create a RegularGridInterpolator
  #      interpolator = RegularGridInterpolator((x, y, z), grid_temperature)

        
        #xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
        #mesh_data = np.meshgrid(*args[0:-1], indexing='ij', sparse=True)
        
        #interp = RegularGridInterpolator((y,x,z), args[-1])
        

    def predict(self, predict_x: np.ndarray[float]) -> np.ndarray[float]:
        """Return n predicted outputs of dimension 1 given inputs.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of d
                dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n predicted 1D values.
        """
        scaled_x = self.prepare_predict(predict_x)
        value = self._regressor(scaled_x).reshape(-1, 1)
        return value.reshape(-1)
######    


class GPModel(RegressionModel):
    """Gaussian process regressor.

    Uses Gaussian process regression. Interpolates & extrapolates. Acts in any
    dimension d >= 1. Learns from any number of training data points n >= 3.
    Time complexity of around O(n^3). Requires hyperparameter optimisation.
    """

    def __init__(self, num_input_dim: int) -> None:
        """Initialise class instance.

        Args:
            num_input_dim (int): number of features (dimensions) for the
                training data.

        Raises:
            Exception: error to explain user's mistake.
        """
        super().__init__(num_input_dim, 3)
        if num_input_dim <= 0:
            raise Exception("Input data should have d >= 1 dimensions.")

    def fit(
        self, train_x: np.ndarray[float], train_y: np.ndarray[float]
    ) -> None:
        """Fit the model to some training data.

        Args:
            train_x (np.ndarray[float]): n by d array of n training inputs with
                d dimensions.
            train_y (np.ndarray[float]): n by 1 array of n training outputs.
        """
        scaled_x = self.prepare_fit(train_x, train_y)
        scaled_x_gp_model = self.prepare_fit(train_x, train_y)
        self._regressor = GaussianProcessRegressor(
            kernel=RBF(), n_restarts_optimizer=10, normalize_y=True
        )
        self._regressor.fit(scaled_x, train_y)

    def predict(self, predict_x: np.ndarray[float]) -> np.ndarray[float]:
        """Return n predicted outputs of dimension 1 given inputs.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of d
                dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n predicted 1D values.
        """
        scaled_x = self.prepare_predict(predict_x)
        return self._regressor.predict(scaled_x).reshape(-1)


class PModel(RegressionModel):
    """Polynomial fit regressor.

    Uses a polynomial fit. Interpolates and extrapolates. Acts in 1D only.
    Learns from any number of training data points n >= degree. Time complexity
    of around O(n^2).
    """

    def __init__(self, num_input_dim: int, degree=3) -> None:
        """Initialise class instance.

        Args:
            num_input_dim (int): number of features (dimensions) for the
                training data.

        Raises:
            Exception: error to explain user's mistake.
        """
        super().__init__(num_input_dim, degree)
        if num_input_dim != 1:
            raise Exception("Input data should have d = 1 dimensions.")
        self._degree = degree

    def fit(
        self, train_x: np.ndarray[float], train_y: np.ndarray[float]
    ) -> None:
        """Fit the model to some training data.

        Args:
            train_x (np.ndarray[float]): n by d array of n training inputs with
                d dimensions.
            train_y (np.ndarray[float]): n by 1 array of n training outputs.
        """
        scaled_x = self.prepare_fit(train_x, train_y)
        self._regressor = np.polyfit(scaled_x[:,0], train_y, deg=self._degree)
        )

    def predict(self, predict_x: np.ndarray[float]) -> np.ndarray[float]:
        """Return n predicted outputs of dimension 1 given inputs.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of d
                dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n predicted 1D values.
        """
        scaled_x = self.prepare_predict(predict_x)
        return self._regressor(scaled_x).reshape(-1, 1)


class CSModel(RegressionModel):
    """Cubic spline regressor.
    
    Uses cubic spline interpolation. Interpolates and extrapolates. Acts in
    1D only. Learns from any number of training data points n >= 2. Time
    complexity of around O(n).
    """

    def __init__(self, num_input_dim: int) -> None:
        """Initialise class instance.

        Args:
            num_input_dim (int): number of features (dimensions) for the
                training data.

        Raises:
            Exception: error to explain user's mistake.
        """
        super().__init__(num_input_dim, 2)
        if num_input_dim != 1:
            raise Exception("Input data should have d = 1 dimensions.")

    def fit(self, train_x: np.ndarray[float], train_y: np.ndarray[float]) -> None:
        """Fit the model to some training data.

        Args:
            train_x (np.ndarray[float]): n by d array of n training inputs with
                d dimensions.
            train_y (np.ndarray[float]): n by 1 array of n training outputs.
        """
        self, train_x: np.ndarray[float], train_y: np.ndarray[float]
    ) -> None:
        """Fit the model to some training data.

        Args:
            train_x (np.ndarray[float]): n by d array of n training inputs with
                d dimensions.
            train_y (np.ndarray[float]): n by 1 array of n training outputs.
        """
        scaled_x = self.prepare_fit(train_x, train_y)
        pos_val_matrix = np.concatenate(
            (scaled_x, train_y.reshape(-1, 1)), axis=1
        )
        pos_val_matrix = pos_val_matrix[pos_val_matrix[:, 0].argsort()]

        self._regressor = CubicSpline(
            pos_val_matrix[:, 0], pos_val_matrix[:, 1]
        )

    def predict(self, predict_x: np.ndarray[float]) -> np.ndarray[float]:
        """Return n predicted outputs of dimension 1 given inputs.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of d
                dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n predicted 1D values.
        """
        scaled_x = self.prepare_predict(predict_x)
        return self._regressor(scaled_x).reshape(-1, 1)


class CTModel(RegressionModel):
    """Clough Tocher regressor.

    Uses a Clough Tocher interpolation. Interpolates only. Acts in 2D only,
    Learns from any number of training data points n >= 3. Time complexity of
    around O(n log(n)) due to the triangulation involved.
    """

    def __init__(self, num_input_dim: int) -> None:
        """Initialise class instance.

        Args:
            num_input_dim (int): number of features (dimensions) for the
                training data.

        Raises:
            Exception: error to explain user's mistake.
        """
        super().__init__(num_input_dim, 3)
        if num_input_dim != 2:
            raise Exception("Input data should have d = 2 dimensions.")
        self._output_mean = 0

    def fit(
        self, train_x: np.ndarray[float], train_y: np.ndarray[float]
    ) -> None:
        """Fit the model to some training data.

        Args:
            train_x (np.ndarray[float]): n by d array of n training inputs with
                d dimensions.
            train_y (np.ndarray[float]): n by 1 array of n training outputs.
        """
        scaled_x = self.prepare_fit(train_x, train_y)
        self._regressor = CloughTocher2DInterpolator(
            scaled_x, train_y, fill_value=np.mean(train_y)
        )

    def predict(self, predict_x: np.ndarray[float]) -> None:
        """Return n predicted outputs of dimension 1 given inputs.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of d
                dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n predicted 1D values.
        """
        scaled_x = self.prepare_predict(predict_x)
        value = self._regressor(scaled_x).reshape(-1, 1)
        return value
