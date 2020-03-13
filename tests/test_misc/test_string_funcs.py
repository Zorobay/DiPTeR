import unittest

from src.misc import string_funcs


class TestStringFuncs(unittest.TestCase):

    def test_split_on_upper_case(self):
        msg = "Failed to split string on upper case!"
        self.assertListEqual(string_funcs.split_on_upper_case("bestParentEver"), ["best", "Parent", "Ever"], msg)
        self.assertListEqual(string_funcs.split_on_upper_case("bestparentEver"), ["bestparent", "Ever"], msg)
        self.assertListEqual(string_funcs.split_on_upper_case("BestParentEver"), ["Best", "Parent", "Ever"], msg)
        self.assertListEqual(string_funcs.split_on_upper_case("BestParent Ever"), ["Best", "Parent", "Ever"], msg)
        self.assertListEqual(string_funcs.split_on_upper_case("BestParent Ever", strip=False), ["Best", "Parent ", "Ever"], msg)
        self.assertListEqual(string_funcs.split_on_upper_case("bestparent ever!"), ["bestparent ever!"], msg)

    def test_snake_case_to_names(self):
        s = "snake_case_is_great"
        ns = string_funcs.snake_case_to_names(s)
        self.assertListEqual(ns, ["Snake", "Case", "Is", "Great"], "Failed to convert snake case to list of names!")

        s = "SnakeCase-is_Great____hi!"
        ns = string_funcs.snake_case_to_names(s)
        self.assertListEqual(ns, ["SnakeCase-is", "Great", "Hi!"], "Failed to convert more advanced case to list of names!")

        s = "snakecaseisgreat"
        ns = string_funcs.snake_case_to_names(s)
        self.assertListEqual(ns, ["Snakecaseisgreat"], "Word without underscores failed to return a list with a single word capitalized!")
