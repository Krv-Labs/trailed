import numpy as np
import pytest

from dect import EctTransformer, SklearnEctConfig
import dect.ect as ect_module


class GraphData:
    def __init__(self, x, batch=None, edge_index=None, face=None):
        self.x = x
        self.batch = (
            np.asarray(batch, dtype=np.int64)
            if batch is not None
            else np.zeros(x.shape[0], dtype=np.int64)
        )
        self.edge_index = edge_index
        self.face = face


class FakeBackend:
    @staticmethod
    def _base(nh, batch, lin, dim_size, scale):
        out = np.zeros((dim_size, lin.shape[0], nh.shape[1]), dtype=np.float32)
        for node_idx, graph_idx in enumerate(batch):
            out[graph_idx] += (nh[node_idx][None, :] + lin[:, None]) * scale
        return out

    def compute_ect_points_forward(self, nh, batch, lin, dim_size):
        return self._base(nh, batch, lin, dim_size, 1.0)

    def compute_ect_points_derivative_forward(self, nh, batch, lin, dim_size):
        return self._base(nh, batch, lin, dim_size, 0.5)

    def compute_ect_edges_forward(self, nh, batch, edge_index, lin, dim_size):
        return self._base(nh, batch, lin, dim_size, 0.75)

    def compute_ect_faces_forward(
        self, nh, batch, edge_index, face, lin, dim_size
    ):
        return self._base(nh, batch, lin, dim_size, 1.25)


def _make_graphs(n=6, num_nodes=5, num_features=3):
    graphs = []
    rng = np.random.default_rng(7)
    for _ in range(n):
        x = rng.normal(size=(num_nodes, num_features)).astype(np.float32)
        edge_index = np.array(
            [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64
        )
        face = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
        graphs.append(GraphData(x=x, edge_index=edge_index, face=face))
    return graphs


def test_sklearn_config_reproducibility_and_fixed_shape():
    cfg_1 = SklearnEctConfig(
        num_thetas=8, num_features=3, direction_mode="random", random_state=42
    )
    cfg_2 = SklearnEctConfig(
        num_thetas=8, num_features=3, direction_mode="random", random_state=42
    )

    v1 = cfg_1.build_directions()
    v2 = cfg_2.build_directions()
    assert v1.shape == (3, 8)
    assert np.allclose(v1, v2)

    fixed = np.ones((3, 8), dtype=np.float32)
    cfg_fixed = SklearnEctConfig(
        num_thetas=8, num_features=3, direction_mode="fixed", directions=fixed
    )
    v_fixed = cfg_fixed.build_directions()
    assert v_fixed.shape == (3, 8)
    assert np.allclose(np.linalg.norm(v_fixed, axis=0), 1.0)


def test_transform_shape_and_normalization(monkeypatch):
    monkeypatch.setattr(ect_module, "_dect_backend", FakeBackend())
    graphs = _make_graphs(n=4)

    transformer = EctTransformer(
        ect_type="points",
        num_thetas=6,
        bump_steps=5,
        num_features=3,
        normalized=True,
        random_state=10,
    )
    X = transformer.fit_transform(graphs)

    assert X.shape == (4, 30)
    X3 = X.reshape(4, 5, 6)
    assert np.all(np.isfinite(X))
    assert np.allclose(np.max(X3, axis=(1, 2)), 1.0)


@pytest.mark.parametrize(
    "ect_type", ["points", "points_derivative", "edges", "faces"]
)
def test_all_ect_types_return_stable_2d_features(monkeypatch, ect_type):
    monkeypatch.setattr(ect_module, "_dect_backend", FakeBackend())
    graphs = _make_graphs(n=3)
    transformer = EctTransformer(
        ect_type=ect_type,
        num_thetas=4,
        bump_steps=7,
        num_features=3,
        normalized=False,
        random_state=1,
    )
    out = transformer.fit_transform(graphs)
    assert out.shape == (3, 28)
    assert np.all(np.isfinite(out))


def test_pipeline_end_to_end_with_sklearn(monkeypatch):
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    monkeypatch.setattr(ect_module, "_dect_backend", FakeBackend())
    graphs = _make_graphs(n=8)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    pipe = Pipeline(
        [
            (
                "ect",
                EctTransformer(
                    ect_type="points",
                    num_thetas=5,
                    bump_steps=4,
                    num_features=3,
                    random_state=3,
                ),
            ),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    pipe.fit(graphs, y)
    preds = pipe.predict(graphs)
    assert preds.shape == (8,)


def test_transform_rejects_non_graph_samples(monkeypatch):
    monkeypatch.setattr(ect_module, "_dect_backend", FakeBackend())
    transformer = EctTransformer(
        ect_type="points",
        num_thetas=4,
        bump_steps=4,
        num_features=3,
        random_state=2,
    ).fit(_make_graphs(n=1))

    with pytest.raises(ValueError, match="All samples passed to transform"):
        transformer.transform([_make_graphs(n=1)[0], {"x": "not-graph"}])
