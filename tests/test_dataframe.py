"""Tests for DataFrame integration."""

import numpy as np
import pytest

from dect.tabular import compute_ect_from_numpy


class TestComputeEctFromNumpy:
    def test_single_point_cloud(self):
        points = np.random.randn(100, 3).astype(np.float32)
        
        ect = compute_ect_from_numpy(
            points,
            num_thetas=16,
            resolution=16,
        )
        
        assert ect.shape == (16, 16)
    
    def test_grouped_point_clouds(self):
        points = np.random.randn(100, 3).astype(np.float32)
        group_ids = np.repeat(np.arange(4), 25)
        
        ect = compute_ect_from_numpy(
            points,
            group_ids=group_ids,
            num_thetas=16,
            resolution=16,
        )
        
        assert ect.shape == (4, 16, 16)
    
    def test_with_channels(self):
        points = np.random.randn(100, 3).astype(np.float32)
        channel_ids = np.random.randint(0, 3, 100)
        
        ect = compute_ect_from_numpy(
            points,
            channel_ids=channel_ids,
            num_thetas=16,
            resolution=16,
        )
        
        assert ect.shape == (16, 16, 3)
    
    def test_grouped_with_channels(self):
        points = np.random.randn(100, 3).astype(np.float32)
        group_ids = np.repeat(np.arange(4), 25)
        channel_ids = np.random.randint(0, 3, 100)
        
        ect = compute_ect_from_numpy(
            points,
            group_ids=group_ids,
            channel_ids=channel_ids,
            num_thetas=16,
            resolution=16,
        )
        
        assert ect.shape == (4, 16, 16, 3)
    
    def test_different_dimensions(self):
        for d in [2, 3, 5]:
            points = np.random.randn(50, d).astype(np.float32)
            
            ect = compute_ect_from_numpy(
                points,
                num_thetas=8,
                resolution=8,
            )
            
            assert ect.shape == (8, 8)
    
    def test_normalized(self):
        points = np.random.randn(100, 3).astype(np.float32)
        
        ect = compute_ect_from_numpy(
            points,
            num_thetas=16,
            resolution=16,
            normalized=True,
        )
        
        assert ect.max() <= 1.0 + 1e-5


class TestPandasIntegration:
    @pytest.fixture
    def df(self):
        pd = pytest.importorskip("pandas")
        return pd.DataFrame({
            "x": np.random.randn(100),
            "y": np.random.randn(100),
            "z": np.random.randn(100),
            "group": np.repeat(range(10), 10),
            "atom_type": np.random.randint(0, 3, 100),
        })
    
    def test_compute_ect_from_pandas(self, df):
        from dect.tabular import compute_ect_from_pandas
        
        ect = compute_ect_from_pandas(
            df,
            coord_columns=["x", "y", "z"],
            num_thetas=16,
            resolution=16,
        )
        
        assert ect.shape == (16, 16)
    
    def test_compute_ect_from_pandas_grouped(self, df):
        from dect.tabular import compute_ect_from_pandas
        
        ect = compute_ect_from_pandas(
            df,
            coord_columns=["x", "y", "z"],
            group_column="group",
            num_thetas=16,
            resolution=16,
        )
        
        assert ect.shape == (10, 16, 16)
    
    def test_compute_ect_from_pandas_with_channels(self, df):
        from dect.tabular import compute_ect_from_pandas
        
        ect = compute_ect_from_pandas(
            df,
            coord_columns=["x", "y", "z"],
            group_column="group",
            channel_column="atom_type",
            num_thetas=16,
            resolution=16,
        )
        
        assert ect.shape == (10, 16, 16, 3)


class TestPolarsIntegration:
    @pytest.fixture
    def df(self):
        pl = pytest.importorskip("polars")
        return pl.DataFrame({
            "x": np.random.randn(100),
            "y": np.random.randn(100),
            "z": np.random.randn(100),
            "group": np.repeat(range(10), 10),
            "atom_type": np.random.randint(0, 3, 100),
        })
    
    def test_compute_ect_from_polars(self, df):
        from dect.tabular import compute_ect_from_polars
        
        ect = compute_ect_from_polars(
            df,
            coord_columns=["x", "y", "z"],
            num_thetas=16,
            resolution=16,
        )
        
        assert ect.shape == (16, 16)
    
    def test_compute_ect_from_polars_grouped(self, df):
        from dect.tabular import compute_ect_from_polars
        
        ect = compute_ect_from_polars(
            df,
            coord_columns=["x", "y", "z"],
            group_column="group",
            num_thetas=16,
            resolution=16,
        )
        
        assert ect.shape == (10, 16, 16)


class TestEctToDataframe:
    def test_to_pandas(self):
        pd = pytest.importorskip("pandas")
        from dect.tabular import ect_to_dataframe
        
        ect = np.random.randn(5, 8, 8).astype(np.float32)
        df = ect_to_dataframe(ect, as_polars=False)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (5, 64)
    
    def test_to_polars(self):
        pl = pytest.importorskip("polars")
        from dect.tabular import ect_to_dataframe
        
        ect = np.random.randn(5, 8, 8).astype(np.float32)
        df = ect_to_dataframe(ect, as_polars=True)
        
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (5, 64)
    
    def test_with_group_ids(self):
        pd = pytest.importorskip("pandas")
        from dect.tabular import ect_to_dataframe
        
        ect = np.random.randn(5, 8, 8).astype(np.float32)
        group_ids = ["A", "B", "C", "D", "E"]
        df = ect_to_dataframe(ect, group_ids=group_ids, as_polars=False)
        
        assert list(df.index) == group_ids


class TestDataFrameEctTransformer:
    def test_pandas(self):
        pd = pytest.importorskip("pandas")
        from dect.tabular import DataFrameEctTransformer
        
        df = pd.DataFrame({
            "x": np.random.randn(100),
            "y": np.random.randn(100),
            "z": np.random.randn(100),
            "group": np.repeat(range(10), 10),
        })
        
        transformer = DataFrameEctTransformer(
            coord_columns=["x", "y", "z"],
            group_column="group",
            num_thetas=16,
            resolution=16,
            output_format="numpy",
        )
        
        ect = transformer.fit_transform(df)
        
        assert ect.shape == (10, 16, 16)
    
    def test_output_format_pandas(self):
        pd = pytest.importorskip("pandas")
        from dect.tabular import DataFrameEctTransformer
        
        df = pd.DataFrame({
            "x": np.random.randn(100),
            "y": np.random.randn(100),
            "z": np.random.randn(100),
            "group": np.repeat(range(10), 10),
        })
        
        transformer = DataFrameEctTransformer(
            coord_columns=["x", "y", "z"],
            group_column="group",
            num_thetas=8,
            resolution=8,
            output_format="pandas",
        )
        
        result = transformer.fit_transform(df)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 64)
