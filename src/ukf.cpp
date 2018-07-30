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

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2;

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
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  is_initialized_ = false;

  n_x_ = 5;

  n_aug_ = 7;

  lambda_ = 3 - n_aug_;

  weights_ = VectorXd(2 * n_aug_ + 1);

  n_z_ = 3;
  z_pred_ = VectorXd(n_z_);
  z_pred_.fill(0.0);
  MatrixXd S_ = MatrixXd(n_z_, n_z_);
  //matrix for sigma points in measurement space
  Zsig_pred_ = MatrixXd(n_z_, 2 * n_aug_ + 1);

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
  x_aug = VectorXd::Zero(n_aug);
  x_aug.head(n_x) = x_;
  //create augmented covariance matrix
  int q_dim = n_aug_ - n_x_;
  MatrixXd Q = MatrixXd(q_dim, q_dim);
  Q <<  std_a_ * std_a_, 0,
        0, std_yawdd_ * std_yawdd_;
  //create square root matrix
  P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(q_dim, q_dim) = Q;
  MatrixXd A = P_aug.llt().matrixL();
  //create augmented sigma points
  MatrixXd sig1 = sqrt(lambda_ + n_aug_) * A;
  sig1.colwise() += x_aug;
  MatrixXd sig2 = sqrt(lambda_ + n_aug_) * A;
  sig2.colwise() -= x_aug;
  //set sigma points as columns of matrix Xsig
  Xsig_aug.col(0) = x_aug;
  Xsig_aug.block(0, 1, n_aug_, n_aug_) = sig1;
  Xsig_aug.block(0, n_aug_ + 1, n_aug_, n_aug_) -= sig2;

  //print result
  //std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;

  //write result
  *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(MatrixXd& Xsig_aug, double delta_t) {
  //create matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  //predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
      VectorXd x = Xsig_aug.block(0, i, n_x_, 1);
      float nu_a = Xsig_aug(n_x_, i);
      float nu_yawdd = Xsig_aug(n_x_ + 1, i);
      float v = x(n_x_ - 3);
      float yaw = x(n_x_ - 2);
      float yawd = x(n_x_ - 1);
      VectorXd noise = VectorXd(n_x_);
      double half_dt2 = 0.5 * delta_t * delta_t;
      noise << half_dt2 * cos(yaw) * nu_a,
               half_dt2 * sin(yaw) * nu_a,
               delta_t * nu_a,
               half_dt2 * nu_yawdd,
               delta_t * nu_yawdd;
      //avoid division by zero
      VectorXd x_dt = VectorXd(n_x_);
      if (fabs(yawd) > 0.001) {
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
      Xsig_pred_.block(0, i, n_x, 1) = x + x_dt + noise;
  }
  //print result
  // std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;
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
      Xsig_diff(1) = fmod(Xsig_diff(1), 2 * M_PI);
      P += weights_(i) * Xsig_diff * Xsig_diff.transpose();
  }

  //print result
  // std::cout << "Predicted state" << std::endl;
  // std::cout << x << std::endl;
  // std::cout << "Predicted covariance matrix" << std::endl;
  // std::cout << P << std::endl;

  //write result
  x_ = x;
  P_ = P;
}

void UKF::PredictRadarMeasurement() {
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
      VectorXd x = Xsig_pred_.col(i);
      float p_x = x(0);
      float p_y = x(1);
      float v = x(2);
      float yaw = x(3);
      float yawd = x(4);
      //transform sigma points into measurement space
      float rho = sqrt(p_x * p_x + p_y * p_y);
      float phi = M_PI / 2;
      float rhod = 10000;
      if (fabs(p_x) > 0.0001) {
        phi = atan2(p_y, p_x);
        phi = fmod(phi, 2 * M_PI);
      }
      else if (p_y < 0) {
        phi *= -1;
      }
      if (fabs(rho) > 0.0001) {
          rhod = (p_x * cos(yaw) * v + p_y * sin(yaw) * v) / rho;
      }
      // Assuming measurement noise w_k_1 = 0
      Zsig_pred_.col(i) << rho, phi, rhod;
      //calculate mean predicted measurement
      z_pred_ += weights_(i) * Zsig_pred_.col(i);
  }
  //calculate innovation covariance matrix S
  S_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
      VectorXd Zsig_diff = Zsig_pred_.col(i) - z_pred_;
      // Normalize angle
      Zsig_diff(1) = fmod(Zsig_diff(1), 2 * M_PI);
      S_ += weights_(i) * Zsig_diff * Zsig_diff.transpose();
  }
  MatrixXd R = MatrixXd(n_z_, n_z_);
  R << std_radr_ * std_radr_, 0, 0,
       0, std_radphi_ * std_radphi_, 0,
       0, 0, std_radrd_ * std_radrd_;
  S_ += R;

  //print result
  // std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  // std::cout << "S: " << std::endl << S << std::endl;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_) {
    float p_x = 0, p_y = 0;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
       * Convert radar from polar to cartesian coordinates.
       */
      float rho = meas_package.raw_measurements_(0);
      float phi = meas_package.raw_measurements_(1);
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
    x_ << p_x, p_y, 0.1, M_PI / 8, 0.01;
    time_us_ = meas_package.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
  //compute the time elapsed between the current and previous measurements
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
  time_us_ = meas_package.timestamp_;
  // Predict
  Prediction(dt);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

    PredictRadarMeasurement();
    // Radar updates
    UpdateRadar(meas_package.raw_measurements_);
  }
  else {
    // Laser updates
    UpdateLidar(meas_package.raw_measurements_);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  AugmentedSigmaPoints(&Xsig_aug);
  SigmaPointPrediction(delta_t);
  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

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
      Xsig_diff(1) = fmod(Xsig_diff(1), 2 * M_PI);
      VectorXd Zsig_diff = Zsig_pred_.col(i) - z_pred_;
      // Normalize angle
      Zsig_diff(1) = fmod(Zsig_diff(1), 2 * M_PI);
      Tc += weights(i) * Xsig_diff * Zsig_diff.transpose();
  }
  //calculate Kalman gain K;
  MatrixXd K = Tc * S_.inverse();
  //update state mean and covariance matrix
  //residual
  VectorXd z_diff = z - z_pred_;
  // Normalize angle
  z_diff(1) = fmod(z_diff(1), 2 * M_PI);
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();
}
