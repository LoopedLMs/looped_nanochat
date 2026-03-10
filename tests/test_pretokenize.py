"""
Tests for pack_batch from pretokenize.py.

python -m pytest tests/test_pretokenize.py -v
"""

import pytest

from scripts.pretokenize import pack_batch


class TestPackBatch:
    """Test the BOS-aligned best-fit packing algorithm."""

    def test_basic_packing(self):
        """Multiple small docs get packed into a single row."""
        doc_buffer = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
        rows = pack_batch(doc_buffer, row_capacity=10, rows_per_batch=1)
        assert len(rows) == 1
        assert len(rows[0]) == 10

    def test_empty_buffer(self):
        """Empty buffer returns no rows."""
        rows = pack_batch([], row_capacity=10, rows_per_batch=5)
        assert rows == []

    def test_exact_fit(self):
        """Doc that exactly fills a row should produce that row with no cropping."""
        doc = list(range(10))
        doc_buffer = [doc]
        rows = pack_batch(doc_buffer, row_capacity=10, rows_per_batch=1)
        assert len(rows) == 1
        assert rows[0] == list(range(10))
        assert doc_buffer == []  # buffer should be consumed

    def test_doc_larger_than_capacity_gets_cropped(self):
        """A doc larger than row_capacity should be cropped to fill the row."""
        doc_buffer = [list(range(20))]
        rows = pack_batch(doc_buffer, row_capacity=10, rows_per_batch=1)
        assert len(rows) == 1
        assert len(rows[0]) == 10
        assert rows[0] == list(range(10))  # first 10 tokens

    def test_single_doc_smaller_than_capacity(self):
        """Single doc smaller than capacity: it's placed, then since buffer is
        empty the loop exits and the row assertion fails (row won't be full).
        pack_batch requires enough docs to fill each row."""
        # With only 1 small doc and nothing else, the while loop will exit
        # when doc_buffer is empty and pos < row_capacity, causing assertion error
        doc_buffer = [[1, 2, 3]]
        with pytest.raises(AssertionError):
            pack_batch(doc_buffer, row_capacity=10, rows_per_batch=1)

    def test_multiple_rows(self):
        """Produces the requested number of rows when buffer has enough docs."""
        # 20 docs of length 5, capacity 10 -> each row fits 2 docs -> 10 rows possible
        doc_buffer = [list(range(5)) for _ in range(20)]
        rows = pack_batch(doc_buffer, row_capacity=10, rows_per_batch=4)
        assert len(rows) == 4
        assert all(len(r) == 10 for r in rows)

    def test_bestfit_picks_largest_fitting(self):
        """Best-fit should prefer the largest doc that fits entirely."""
        # Row capacity = 10, docs: [3 tokens], [7 tokens], [5 tokens]
        # First pick: largest fitting = [7 tokens], remaining = 3
        # Then: [3 tokens] fits exactly
        doc_buffer = [[1, 1, 1], [2, 2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3]]
        rows = pack_batch(doc_buffer, row_capacity=10, rows_per_batch=1)
        assert len(rows) == 1
        assert len(rows[0]) == 10
        # Should start with the 7-token doc, then the 3-token doc
        assert rows[0][:7] == [2, 2, 2, 2, 2, 2, 2]
        assert rows[0][7:] == [1, 1, 1]

    def test_buffer_consumed_in_place(self):
        """pack_batch modifies doc_buffer in place, removing consumed docs."""
        doc_buffer = [list(range(5)) for _ in range(10)]
        pack_batch(doc_buffer, row_capacity=10, rows_per_batch=2)
        # 2 rows * 2 docs per row = 4 docs consumed
        assert len(doc_buffer) == 6

    def test_crops_shortest_when_nothing_fits(self):
        """When no doc fits the remaining space, crops the shortest doc."""
        # Row capacity = 10, one doc of 8 tokens, one of 5 tokens
        # First: 8-token doc fits, remaining = 2
        # Nothing fits entirely -> crop shortest (5-token) to fill 2
        doc_buffer = [[1] * 8, [2] * 5]
        rows = pack_batch(doc_buffer, row_capacity=10, rows_per_batch=1)
        assert len(rows) == 1
        assert rows[0] == [1] * 8 + [2] * 2
