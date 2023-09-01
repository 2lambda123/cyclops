import sys
import os

# Sort the paths out to run from this file
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(os.path.sep,parent_path, 'src')
sys.path.append(src_path)

from sensors2 import Sensor, PointSensor, RoundSensor, MultiSensor, Thermocouple
import numpy as np
import unittest




class Test(unittest.TestCase):
    def test_sensor_scalar(self):
        def f(x): return np.zeros(x.shape)
        sensor = Sensor(0, f, 0.4, np.array([[-500], [500]]), np.array([[0, 0], [1, 1], [-1, -1]]))
        
        relative_pos = sensor.get_sites()
        actual_pos = np.array([5, 5])
        measure_pos = relative_pos + actual_pos*np.ones(relative_pos.shape)
        true_pos = np.array([[5, 5], [6, 6], [4, 4]])
        for i, pos in enumerate(measure_pos):
            self.assertEqual(pos[0], true_pos[i, 0])

        true_values = np.array([[1], [2], [0]])
        sensor.set_values(true_values)
        sensor_value = sensor.get_values()
        self.assertEqual(sensor_value[0, 0], 1)

        def f(x): return np.zeros(x.shape)
        sensor = Sensor(0, f, 0.4, np.array([[-0.1], [0.1]]), np.array([[0, 0], [1, 1], [-1, -1]]))
        
        relative_pos = sensor.get_sites()
        actual_pos = np.array([5, 5])
        measure_pos = relative_pos + actual_pos*np.ones(relative_pos.shape)
        true_pos = np.array([[5, 5], [6, 6], [4, 4]])
        for i, pos in enumerate(measure_pos):
            self.assertEqual(pos[0], true_pos[i, 0])

        true_values = np.array([[1], [2], [0]])
        sensor.set_values(true_values)
        sensor_value = sensor.get_values()
        self.assertEqual(sensor_value[0, 0], 0.1)


    def test_sensor_vector(self):
        def f(x): return np.zeros(x.shape)
        sensor = Sensor(0, f, 0.4, np.array([[-5, -5], [5, 5]]), np.array([[0, 0], [1, 1], [-1, -1]]))
        
        relative_pos = sensor.get_sites()
        actual_pos = np.array([5, 5])
        measure_pos = relative_pos + actual_pos*np.ones(relative_pos.shape)
        true_pos = np.array([[5, 5], [6, 6], [4, 4]])
        for i, pos in enumerate(measure_pos):
            self.assertEqual(pos[0], true_pos[i, 0])

        true_values = np.array([[1, 1], [2, 2], [0, 3]])
        sensor.set_values(true_values)
        sensor_value = sensor.get_values()
        self.assertEqual(sensor_value[0, 0], 1)
        self.assertEqual(sensor_value[0, 1], 2)


    def test_point_1D(self):
        def f(x): return np.zeros(x.shape)
        sensor = PointSensor(0, f, 0.4, np.array([[-5], [5]]), 1)
        
        relative_pos = sensor.get_sites()
        actual_pos = np.array([5])
        measure_pos = relative_pos + actual_pos*np.ones(relative_pos.shape)
        true_pos = np.array([[5]])
        for i, pos in enumerate(measure_pos):
            self.assertEqual(pos[0], true_pos[i, 0])

        true_values = np.array([[1]])
        sensor.set_values(true_values)
        sensor_value = sensor.get_values()
        self.assertEqual(sensor_value[0, 0], 1)

    
    def test_point_2D(self):
        def f(x): return np.zeros(x.shape)
        sensor = PointSensor(0, f, 0.4, np.array([[-5], [5]]), 2)
        
        relative_pos = sensor.get_sites()
        actual_pos = np.array([5, 5])
        measure_pos = relative_pos + actual_pos*np.ones(relative_pos.shape)
        true_pos = np.array([[5, 5]])
        for i, pos in enumerate(measure_pos):
            self.assertEqual(pos[0], true_pos[i, 0])
            self.assertEqual(pos[1], true_pos[i, 1])

        true_values = np.array([[1]])
        sensor.set_values(true_values)
        sensor_value = sensor.get_values()
        self.assertEqual(sensor_value[0, 0], 1)

    
    def test_round_1D(self):
        def f(x): return np.zeros(x.shape)
        sensor = RoundSensor(0, f, 0.4, np.array([[-5], [5]]), 1, 1)
        
        relative_pos = sensor.get_sites()
        actual_pos = np.array([5])
        measure_pos = relative_pos + actual_pos*np.ones(relative_pos.shape)
        true_pos = np.array([[5], [5], [5], [4], [6]])
        for i, pos in enumerate(measure_pos):
            self.assertEqual(pos[0], true_pos[i, 0])

        true_values = np.array([[1], [2], [3], [4], [5]])
        sensor.set_values(true_values)
        sensor_value = sensor.get_values()
        self.assertEqual(sensor_value[0, 0], 3)
    

    def test_round_2D(self):
        def f(x): return np.zeros(x.shape)
        sensor = RoundSensor(0, f, 0.4, np.array([[-5], [5]]), 1, 2)
        
        relative_pos = sensor.get_sites()
        actual_pos = np.array([5, 5])
        measure_pos = relative_pos + actual_pos*np.ones(relative_pos.shape)
        true_pos = np.array([[5, 5], [5, 6], [5, 4], [4, 5], [6, 5]])
        for i, pos in enumerate(measure_pos):
            self.assertEqual(pos[0], true_pos[i, 0])
            self.assertEqual(pos[1], true_pos[i, 1])

        true_values = np.array([[1]])
        sensor.set_values(true_values)
        sensor_value = sensor.get_values()
        self.assertEqual(sensor_value[0, 0], 1)


    def test_multi_2D(self):
        def f(x): return np.zeros(x.shape)
        sensor = MultiSensor(0, f, 0.4, np.array([[-5], [5]]), np.array([[0, 0], [0, 1], [0, 2]]))

        relative_pos = sensor.get_sites()
        actual_pos = np.array([0, 0])
        measure_pos = relative_pos + actual_pos*np.ones(relative_pos.shape)
        true_pos = np.array([[0, 0], [0, 1], [0, 2]])
        for i, pos in enumerate(measure_pos):
            self.assertEqual(pos[0], true_pos[i, 0])
            self.assertEqual(pos[1], true_pos[i, 1])

        true_values = np.array([[1], [2], [3]])
        sensor.set_values(true_values)
        sensor_value = sensor.get_values()
        
        for i, value in enumerate(sensor_value):
            self.assertEqual(value[0], true_values[i, 0])


    def test_multi_1D(self):
        def f(x): return np.zeros(x.shape)
        sensor = MultiSensor(0, f, 0.4, np.array([[-5], [5]]), np.array([[0], [1], [2]]))

        relative_pos = sensor.get_sites()
        actual_pos = np.array([0, 0])
        measure_pos = relative_pos + actual_pos*np.ones(relative_pos.shape)
        true_pos = np.array([[0], [1], [2]])
        for i, pos in enumerate(measure_pos):
            self.assertEqual(pos[0], true_pos[i, 0])

        true_values = np.array([[1], [2], [3]])
        sensor.set_values(true_values)
        sensor_value = sensor.get_values()
        
        for i, value in enumerate(sensor_value):
            self.assertEqual(value[0], true_values[i, 0])


    def test_thermo_1D(self):
        temps = np.linspace(-10, 10, 20).reshape(-1, 1)
        voltages = temps.copy()
        sensor = Thermocouple(temps, voltages, 1)

        relative_pos = sensor.get_sites()
        actual_pos = np.array([5])
        measure_pos = relative_pos + actual_pos*np.ones(relative_pos.shape)
        true_pos = np.array([[5], [5], [5], [4.99925], [5.00075]])
        for i, pos in enumerate(measure_pos):
            self.assertEqual(pos[0], true_pos[i, 0])

        true_values = np.array([[1], [2], [3], [4], [5]])
        sensor.set_values(true_values)
        sensor_value = sensor.get_values()
        self.assertAlmostEqual(sensor_value[0, 0], 3, -3)

    
    
    def test_thermo_2D(self):
        temps = np.linspace(-10, 10, 20).reshape(-1, 1)
        voltages = temps.copy()
        sensor = Thermocouple(temps, voltages, 2)

        relative_pos = sensor.get_sites()
        actual_pos = np.array([5, 5])
        measure_pos = relative_pos + actual_pos*np.ones(relative_pos.shape)
        true_pos = np.array([[5, 5], [5, 5.00075], [5, 4.99925], [4.99925, 5], [5.00075, 5]])
        for i, pos in enumerate(measure_pos):
            self.assertEqual(pos[0], true_pos[i, 0])

        true_values = np.array([[1], [2], [3], [4], [5]])
        sensor.set_values(true_values)
        sensor_value = sensor.get_values()
        self.assertAlmostEqual(sensor_value[0, 0], 3, -3)
    


if __name__ == "__main__":
    unittest.main()