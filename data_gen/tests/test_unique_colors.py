import unittest

import numpy as np

from vidgeom.assets import assign_unique_token_colors


class UniqueColorFallbackTests(unittest.TestCase):
    def test_generates_colors_beyond_palette(self) -> None:
        tokens = [str(i) for i in range(32)]
        rng = np.random.default_rng(0)
        color_map = assign_unique_token_colors(tokens, rng)
        self.assertEqual(len(color_map), len(tokens))
        primaries = {colors[0] for colors in color_map.values()}
        self.assertEqual(len(primaries), len(tokens))


if __name__ == "__main__":
    unittest.main()
