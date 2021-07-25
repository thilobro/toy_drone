from toy_drone.extended_kalman_filter import ExtendedKalmanFilter
import casadi as ca


def test_constructor():
    test_kalman_filter = ExtendedKalmanFilter(ca.vertcat(0), ca.vertcat(0),
                                              ca.vertcat(1), ca.vertcat(1),
                                              lambda x, u: ca.vertcat(x + u),
                                              lambda x, u: ca.vertcat(x),
                                              lambda x, u: ca.vertcat(1),
                                              lambda x, u: ca.vertcat(1), 1)
    assert 0 == test_kalman_filter.get_estimate()


def test_input_measurement():
    test_kalman_filter = ExtendedKalmanFilter(ca.vertcat(0), ca.vertcat(0),
                                              ca.vertcat(1), ca.vertcat(1),
                                              lambda x, u: ca.vertcat(x + u),
                                              lambda x, u: ca.vertcat(x),
                                              lambda x, u: ca.vertcat(1),
                                              lambda x, u: ca.vertcat(1), 1)
    test_kalman_filter.input_measurement(0)
    assert 0 == test_kalman_filter.get_estimate()
    test_kalman_filter.input_measurement(1)
    assert 0 < test_kalman_filter.get_estimate()


def test_input_controls():
    test_kalman_filter = ExtendedKalmanFilter(ca.vertcat(0), ca.vertcat(0),
                                              ca.vertcat(1), ca.vertcat(1),
                                              lambda x, u: ca.vertcat(x + u),
                                              lambda x, u: ca.vertcat(x),
                                              lambda x, u: ca.vertcat(1),
                                              lambda x, u: ca.vertcat(1), 1)
    test_kalman_filter.input_controls(0)
    test_kalman_filter.input_measurement(0)
    assert 0 == test_kalman_filter.get_estimate()
