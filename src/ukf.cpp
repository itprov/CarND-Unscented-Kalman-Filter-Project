#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  // P_ = MatrixXd(5, 5);
  // Try identity matrix to init P
  P_ = MatrixXd::Identity(5, 5);
  // Experiment with diagonal values other than 1
  // 0.35 seems to work best
  P_ *= 0.33;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

  /**
  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  is_initialized_ = false;

  n_x_ = 5;

  n_aug_ = 7;

  lambda_ = 3 - n_aug_;

  weights_ = VectorXd(2 * n_aug_ + 1);

  k_ = 0;
}

UKF::~UKF() {}

void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;
  //create augmented covariance matrix
  int q_dim = n_aug_ - n_x_;
  MatrixXd Q = MatrixXd(q_dim, q_dim);
  Q <<  std_a_ * std_a_, 0,
        0, std_yawdd_ * std_yawdd_;
  //create square root matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(q_dim, q_dim) = Q;
  MatrixXd sqrt_P_aug = P_aug.llt().matrixL();
  //create augmented sigma points
  double sqrt_coeff = sqrt(lambda_ + n_aug_);
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
		Xsig_aug.col(i+1) = x_aug + sqrt_coeff * sqrt_P_aug.col(i);
		Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt_coeff * sqrt_P_aug.col(i);
	}
  //write result
  *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(MatrixXd& Xsig_aug, double delta_t) {
  //create matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  //predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
      VectorXd x = Xsig_aug.col(i);
      double v = x(n_x_ - 3);
      double yaw = x(n_x_ - 2);
      double yawd = x(n_x_ - 1);
      double nu_a = x(n_x_);
      double nu_yawdd = x(n_x_ + 1);
      VectorXd noise = VectorXd(n_x_);
      double half_dt2 = 0.5 * delta_t * delta_t;
      noise << half_dt2 * cos(yaw) * nu_a,
               half_dt2 * sin(yaw) * nu_a,
               delta_t * nu_a,
               half_dt2 * nu_yawdd,
               delta_t * nu_yawdd;
      //avoid division by zero
      VectorXd x_dt = VectorXd(n_x_);
      if (fabs(yawd) > 0.0001) {
          x_dt << (v / yawd) * (sin(yaw + yawd * delta_t) - sin(yaw)),
                  (v / yawd) * (-cos(yaw + yawd * delta_t) + cos(yaw)),
                  0,
                  yawd * delta_t,
                  0;
      }
      else {
          x_dt << v * cos(yaw) * delta_t,
                  v * sin(yaw) * delta_t,
                  0,
                  yawd * delta_t,
                  0;
      }
      //write predicted sigma points into right column
      Xsig_pred_.col(i) = x.head(n_x_) + x_dt + noise;
  }
}

void UKF::PredictMeanAndCovariance() {
  //create vector for predicted state
  VectorXd x = VectorXd(n_x_);

  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);

  //set weights
  x.fill(0.0);
  double w = 1 / (lambda_ + n_aug_);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
      if (i == 0) {
        weights_(i) = lambda_ * w;
      }
      else {
        weights_(i) = w / 2;
      }

      //predict state mean
      x += weights_(i) * Xsig_pred_.col(i);
  }
  //predict state covariance matrix
  P.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
      VectorXd Xsig_diff = Xsig_pred_.col(i) - x;
      // Normalize angle
      Xsig_diff(3) = fmod(Xsig_diff(3), 2 * M_PI);
      P += weights_(i) * Xsig_diff * Xsig_diff.transpose();
  }

  //write result
  x_ = x;
  P_ = P;
}

void UKF::PredictMeasurement(MeasurementPackage::SensorType sensor_type) {
  if (sensor_type == MeasurementPackage::RADAR) {
    n_z_ = 3;
  }
  else {
    n_z_ = 2;
  }
  z_pred_ = VectorXd(n_z_);
  z_pred_.fill(0.0);
  S_ = MatrixXd(n_z_, n_z_);
  //matrix for sigma points in measurement space
  Zsig_pred_ = MatrixXd(n_z_, 2 * n_aug_ + 1);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
      VectorXd x = Xsig_pred_.col(i);
      double p_x = x(0);
      double p_y = x(1);
      if (sensor_type == MeasurementPackage::RADAR) {
        double v = x(2);
        double yaw = x(3);
        //transform sigma points into measurement space
        double rho = sqrt(p_x * p_x + p_y * p_y);
        double phi = M_PI / 2;
        double rhod = 0;
        if (fabs(p_x) > 0.001) {
          phi = atan2(p_y, p_x);
          phi = fmod(phi, 2 * M_PI);
        }
        else if (p_y < 0) {
          phi *= -1;
        }
        if (fabs(rho) > 0.001) {
            rhod = (p_x * cos(yaw) + p_y * sin(yaw)) * v / rho;
        }
        // Assuming measurement noise w_k_1 = 0
        Zsig_pred_.col(i) << rho, phi, rhod;
      }
      else {
        Zsig_pred_.col(i) << p_x, p_y;
      }
      //calculate mean predicted measurement
      z_pred_ += weights_(i) * Zsig_pred_.col(i);
  }
  //calculate innovation covariance matrix S
  S_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
      VectorXd Zsig_diff = Zsig_pred_.col(i) - z_pred_;
      if (sensor_type == MeasurementPackage::RADAR) {
        // Normalize angle
        Zsig_diff(1) = fmod(Zsig_diff(1), 2 * M_PI);
      }
      S_ += weights_(i) * Zsig_diff * Zsig_diff.transpose();
  }
  MatrixXd R = MatrixXd(n_z_, n_z_);
  if (sensor_type == MeasurementPackage::RADAR) {
    R << std_radr_ * std_radr_, 0, 0,
         0, std_radphi_ * std_radphi_, 0,
         0, 0, std_radrd_ * std_radrd_;
  }
  else {
    // LIDAR R
    R << std_laspx_ * std_laspx_, 0,
         0, std_laspy_ * std_laspy_;
  }
  S_ += R;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_) {
    double p_x = 0, p_y = 0;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
       * Convert radar from polar to cartesian coordinates.
       */
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      p_x = rho * cos(phi);
      p_y = rho * sin(phi);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
       * Get cartesian coordinates.
       */
      p_x = meas_package.raw_measurements_(0);
      p_y = meas_package.raw_measurements_(1);
    }
    // Init x
    x_ << p_x, p_y, 0, 0, 0;
    time_us_ = meas_package.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
  // Measurement sample number update - useful in plotting NIS value later
  k_ += 1;
  //compute the time elapsed between the current and previous measurements
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
  time_us_ = meas_package.timestamp_;
  // Predict
  Prediction(dt);
  // Update based on use_laser_ and use_radar_ values.
  // Ignore LIDAR or RADAR values respectively, if these attributes are false
  if ((use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER)
    || (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR)) {
    PredictMeasurement(meas_package.sensor_type_);
    // Update
    Update(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  AugmentedSigmaPoints(&Xsig_aug);
  SigmaPointPrediction(Xsig_aug, delta_t);
  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a measurement.
 * Measurement can be either laser or radar
 * @param {MeasurementPackage} meas_package
 */
void UKF::Update(MeasurementPackage meas_package) {
  /**
  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  //input measurement
  VectorXd z = meas_package.raw_measurements_.head(n_z_);

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_);
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
      VectorXd Xsig_diff = Xsig_pred_.col(i) - x_;
      // Normalize angle
      Xsig_diff(3) = fmod(Xsig_diff(3), 2 * M_PI);
      VectorXd Zsig_diff = Zsig_pred_.col(i) - z_pred_;
      if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        // Normalize angle
        Zsig_diff(1) = fmod(Zsig_diff(1), 2 * M_PI);
      }
      Tc += weights_(i) * Xsig_diff * Zsig_diff.transpose();
  }
  //calculate Kalman gain K;
  MatrixXd S_inv = S_.inverse();
  MatrixXd K = Tc * S_inv;
  //update state mean and covariance matrix
  //residual
  VectorXd z_diff = z - z_pred_;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Normalize angle
    z_diff(1) = fmod(z_diff(1), 2 * M_PI);
  }
  x_ += K * z_diff;
  // Normalize angle
  x_(3) = fmod(x_(3), 2 * M_PI);
  P_ -= K * S_ * K.transpose();
  // calculate NIS value
	double nis = z_diff.transpose() * S_inv * z_diff;
  // Output NIS values to be used for plotting
  /**
  * E.g. Use gnuplot as follows:
  * ./UnscentedKF > log.txt
  * grep -e "^\d" log.txt > nis.dat
  * gnuplot> plot 'nis.dat' with lines, 7.8
  */
  cout << k_ << " " << nis << endl;
}
