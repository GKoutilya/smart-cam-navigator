import os
import tempfile
import unittest

from src.perception.ground_plane import GroundPlaneMapper, compute_homography

IMAGE_POINTS = [(100.0, 50.0), (500.0, 60.0), (520.0, 400.0), (90.0, 380.0)]
WORLD_POINTS = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


class TestGroundPlaneMapper(unittest.TestCase):
    def setUp(self):
        homography = compute_homography(IMAGE_POINTS, WORLD_POINTS)
        self.mapper = GroundPlaneMapper(homography)

    def test_corners_map_to_expected_world_points(self):
        for image_point, expected in zip(IMAGE_POINTS, WORLD_POINTS):
            actual = self.mapper.pixel_to_world(*image_point)
            self.assertAlmostEqual(actual[0], expected[0], places=4)
            self.assertAlmostEqual(actual[1], expected[1], places=4)

    def test_pixel_world_round_trip(self):
        px, py = 250.0, 200.0
        world = self.mapper.pixel_to_world(px, py)
        pixel = self.mapper.world_to_pixel(*world)
        self.assertAlmostEqual(pixel[0], px, places=3)
        self.assertAlmostEqual(pixel[1], py, places=3)

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "homography.npy")
            self.mapper.save(path)
            loaded = GroundPlaneMapper.from_file(path)

            for image_point, expected in zip(IMAGE_POINTS, WORLD_POINTS):
                actual = loaded.pixel_to_world(*image_point)
                self.assertAlmostEqual(actual[0], expected[0], places=4)
                self.assertAlmostEqual(actual[1], expected[1], places=4)

    def test_compute_homography_requires_four_points(self):
        with self.assertRaises(ValueError):
            compute_homography(IMAGE_POINTS[:3], WORLD_POINTS[:3])


if __name__ == '__main__':
    unittest.main()
