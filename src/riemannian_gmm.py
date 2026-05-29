"""Riemannian Gaussian mixture model for unit quaternions on S^3.

Quaternion convention in this file is scalar-first ``(w, x, y, z)``.
SciPy's ``Rotation.as_quat()`` returns scalar-last ``(x, y, z, w)``; convert
explicitly at API boundaries.
"""

import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp


TINY = 1e-12


# ---- quaternion primitives ----


def normalize_quat(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < TINY:
        raise ValueError("Cannot normalize a zero quaternion")
    return q / norm


def hemisphere_align(q, base):
    q = normalize_quat(q)
    base = normalize_quat(base)
    if np.dot(q, base) < 0.0:
        return -q
    return q


def quat_mul(q1, q2):
    w1, x1, y1, z1 = np.asarray(q1, dtype=float)
    w2, x2, y2, z2 = np.asarray(q2, dtype=float)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_conj(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_log(q, base):
    q = hemisphere_align(q, base)
    base = normalize_quat(base)
    q_rel = normalize_quat(quat_mul(quat_conj(base), q))
    v_part = q_rel[1:4]
    v_norm = np.linalg.norm(v_part)
    if v_norm < TINY:
        return np.zeros(3)
    theta = 2.0 * math.atan2(v_norm, q_rel[0])
    return theta * (v_part / v_norm)


def quat_exp(v, base):
    base = normalize_quat(base)
    v = np.asarray(v, dtype=float)
    theta = np.linalg.norm(v)
    if theta < TINY:
        return base.copy()
    axis = v / theta
    q_rel = np.concatenate(([math.cos(theta / 2.0)], math.sin(theta / 2.0) * axis))
    return normalize_quat(quat_mul(base, q_rel))


def quat_dist(q1, q2):
    return float(np.linalg.norm(quat_log(q1, q2)))


def xyzw_to_wxyz(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


def wxyz_to_xyzw(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def rotations_to_wxyz(quaternions):
    if hasattr(quaternions, "as_quat"):
        return xyzw_to_wxyz(quaternions.as_quat()).reshape(1, 4)
    if isinstance(quaternions, list) and quaternions and hasattr(quaternions[0], "as_quat"):
        return np.vstack([xyzw_to_wxyz(q.as_quat()) for q in quaternions])
    return _as_quat_array(quaternions)


def _as_quat_array(quaternions):
    quaternions = np.asarray(quaternions, dtype=float)
    if quaternions.ndim == 1:
        quaternions = quaternions.reshape(1, 4)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("quaternions must have shape (N, 4)")
    return np.vstack([normalize_quat(q) for q in quaternions])


# ---- karcher mean ----


def karcher_mean(quaternions, weights=None, max_iter=50, tol=1e-8):
    quaternions = _as_quat_array(quaternions)
    n_samples = quaternions.shape[0]
    if weights is None:
        weights = np.ones(n_samples) / n_samples
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.shape[0] != n_samples:
            raise ValueError("weights must have shape (N,)")
        weight_sum = np.sum(weights)
        if weight_sum <= TINY:
            raise ValueError("weights must contain positive mass")
        weights = weights / weight_sum

    mu = quaternions[int(np.argmax(weights))].copy()
    for _ in range(max_iter):
        logs = np.vstack([quat_log(q, mu) for q in quaternions])
        v_bar = weights @ logs
        if np.linalg.norm(v_bar) < tol:
            break
        mu = quat_exp(v_bar, mu)
    return normalize_quat(mu)


# ---- RGMM class ----


class RiemannianGMM:
    def __init__(self, n_components, max_iter=50, tol=1e-6, reg=1e-6, random_state=None):
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reg = float(reg)
        self.random_state = random_state

    def fit(self, quaternions):
        quaternions = _as_quat_array(quaternions)
        if self.n_components < 1 or self.n_components > len(quaternions):
            raise ValueError("n_components must be between 1 and the number of samples")

        self._initialize(quaternions)
        self.log_likelihood_history_ = [float(np.sum(self.score_samples(quaternions)))]

        for _ in range(self.max_iter):
            responsibilities = self.predict_proba(quaternions)
            self._m_step(quaternions, responsibilities)
            log_likelihood = float(np.sum(self.score_samples(quaternions)))

            previous = self.log_likelihood_history_[-1]
            if log_likelihood < previous and previous - log_likelihood < 1e-6:
                log_likelihood = previous
            self.log_likelihood_history_.append(log_likelihood)

            denom = max(abs(previous), 1.0)
            if abs(log_likelihood - previous) / denom < self.tol:
                break

        return self

    def predict_proba(self, quaternions):
        quaternions = rotations_to_wxyz(quaternions)
        log_components = self._estimate_log_components(quaternions)
        sample_log_likelihood = logsumexp(log_components, axis=1, keepdims=True)
        return np.exp(log_components - sample_log_likelihood)

    def predict(self, quaternions):
        return np.argmax(self.predict_proba(quaternions), axis=1)

    def score_samples(self, quaternions):
        quaternions = rotations_to_wxyz(quaternions)
        return logsumexp(self._estimate_log_components(quaternions), axis=1)

    def bic(self, quaternions):
        quaternions = rotations_to_wxyz(quaternions)
        n_params = self.n_components * (3 + 6 + 1) - 1
        log_likelihood = float(np.sum(self.score_samples(quaternions)))
        return n_params * math.log(len(quaternions)) - 2.0 * log_likelihood

    def logProb(self, quaternions):
        """Compatibility wrapper returning responsibilities as (K, N)."""

        return self.predict_proba(quaternions).T

    def _initialize(self, quaternions):
        rng = np.random.default_rng(self.random_state)
        means = self._farthest_first_means(quaternions, rng)

        for _ in range(5):
            assignments = self._nearest_assignments(quaternions, means)
            for k in range(self.n_components):
                mask = assignments == k
                if np.any(mask):
                    means[k] = karcher_mean(quaternions[mask])

        assignments = self._nearest_assignments(quaternions, means)
        overall_mean = karcher_mean(quaternions)
        global_scale = np.mean([quat_dist(q, overall_mean) ** 2 for q in quaternions])
        global_scale = max(global_scale, self.reg)

        weights = np.zeros(self.n_components)
        covariances = np.zeros((self.n_components, 3, 3))
        for k in range(self.n_components):
            mask = assignments == k
            weights[k] = max(np.mean(mask), TINY)
            if np.sum(mask) <= 1:
                covariances[k] = global_scale * np.eye(3) + self.reg * np.eye(3)
            else:
                covariances[k] = self._weighted_covariance(quaternions[mask], means[k], None)
        weights /= np.sum(weights)

        self.means_ = np.vstack([normalize_quat(q) for q in means])
        self.covariances_ = covariances
        self.weights_ = weights

    def _farthest_first_means(self, quaternions, rng):
        first = int(rng.integers(len(quaternions)))
        selected = [first]
        while len(selected) < self.n_components:
            distances = np.array(
                [min(quat_dist(q, quaternions[j]) for j in selected) for q in quaternions]
            )
            selected.append(int(np.argmax(distances)))
        return quaternions[selected].copy()

    @staticmethod
    def _nearest_assignments(quaternions, means):
        distances = np.array([[quat_dist(q, mu) for mu in means] for q in quaternions])
        return np.argmin(distances, axis=1)

    def _estimate_log_components(self, quaternions):
        quaternions = _as_quat_array(quaternions)
        log_components = np.zeros((len(quaternions), self.n_components))
        for k in range(self.n_components):
            logs = np.vstack([quat_log(q, self.means_[k]) for q in quaternions])
            log_components[:, k] = np.log(max(self.weights_[k], TINY))
            log_components[:, k] += _log_gaussian(logs, self.covariances_[k])
        return log_components

    def _m_step(self, quaternions, responsibilities):
        n_samples = len(quaternions)
        nk = responsibilities.sum(axis=0)
        old_means = self.means_.copy()
        old_covariances = self.covariances_.copy()

        for k in range(self.n_components):
            if nk[k] <= TINY:
                self.means_[k] = old_means[k]
                self.covariances_[k] = old_covariances[k]
                continue
            self.means_[k] = karcher_mean(quaternions, weights=responsibilities[:, k])
            self.covariances_[k] = self._weighted_covariance(
                quaternions,
                self.means_[k],
                responsibilities[:, k],
            )
        self.weights_ = np.maximum(nk / n_samples, TINY)
        self.weights_ /= np.sum(self.weights_)

    def _weighted_covariance(self, quaternions, mean, weights):
        logs = np.vstack([quat_log(q, mean) for q in quaternions])
        if weights is None:
            covariance = logs.T @ logs / max(len(logs), 1)
        else:
            weights = np.asarray(weights, dtype=float).reshape(-1)
            covariance = (logs.T * weights) @ logs / max(np.sum(weights), TINY)
        covariance = 0.5 * (covariance + covariance.T)
        return covariance + self.reg * np.eye(3)


def _log_gaussian(x, covariance):
    x = np.asarray(x, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    jitter = 0.0
    for _ in range(6):
        try:
            chol = np.linalg.cholesky(covariance + jitter * np.eye(3))
            break
        except np.linalg.LinAlgError:
            jitter = 10.0 * jitter if jitter else 1e-9
    else:
        chol = np.linalg.cholesky(covariance + 1e-3 * np.eye(3))

    solved = np.linalg.solve(chol, x.T)
    mahalanobis = np.sum(solved * solved, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    return -0.5 * (3.0 * math.log(2.0 * math.pi) + logdet + mahalanobis)


# ---- single-chart baseline for tests and demo ----


class _EuclideanGMM:
    def __init__(self, n_components, max_iter=50, tol=1e-6, reg=1e-6, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg
        self.random_state = random_state

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        rng = np.random.default_rng(self.random_state)
        means = x[self._farthest_first(x, rng)].copy()
        covs = np.tile(np.cov(x.T) + self.reg * np.eye(3), (self.n_components, 1, 1))
        weights = np.ones(self.n_components) / self.n_components
        prev = -np.inf
        for _ in range(self.max_iter):
            logp = np.column_stack(
                [np.log(weights[k]) + _log_gaussian(x - means[k], covs[k]) for k in range(self.n_components)]
            )
            sample_ll = logsumexp(logp, axis=1)
            resp = np.exp(logp - sample_ll[:, None])
            nk = resp.sum(axis=0)
            for k in range(self.n_components):
                if nk[k] <= TINY:
                    continue
                means[k] = resp[:, k] @ x / nk[k]
                centered = x - means[k]
                covs[k] = (centered.T * resp[:, k]) @ centered / nk[k] + self.reg * np.eye(3)
            weights = np.maximum(nk / len(x), TINY)
            weights /= np.sum(weights)
            ll = float(np.sum(sample_ll))
            if abs(ll - prev) / max(abs(prev), 1.0) < self.tol:
                break
            prev = ll
        self.means_ = means
        self.covariances_ = covs
        self.weights_ = weights
        return self

    def score_samples(self, x):
        x = np.asarray(x, dtype=float)
        logp = np.column_stack(
            [np.log(self.weights_[k]) + _log_gaussian(x - self.means_[k], self.covariances_[k])
             for k in range(self.n_components)]
        )
        return logsumexp(logp, axis=1)

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        logp = np.column_stack(
            [np.log(self.weights_[k]) + _log_gaussian(x - self.means_[k], self.covariances_[k])
             for k in range(self.n_components)]
        )
        return np.argmax(logp, axis=1)

    def _farthest_first(self, x, rng):
        selected = [int(rng.integers(len(x)))]
        while len(selected) < self.n_components:
            distances = np.array([min(np.linalg.norm(row - x[j]) for j in selected) for row in x])
            selected.append(int(np.argmax(distances)))
        return selected


# ---- tests ----


def _sample_component(mean, covariance, n_samples, rng):
    vectors = rng.multivariate_normal(np.zeros(3), covariance, size=n_samples)
    return np.vstack([quat_exp(v, mean) for v in vectors])


def test_roundtrip():
    rng = np.random.default_rng(10)
    for _ in range(100):
        q = normalize_quat(rng.normal(size=4))
        base = normalize_quat(rng.normal(size=4))
        recovered = quat_exp(quat_log(q, base), base)
        assert quat_dist(q, recovered) < 1e-8


def test_karcher():
    rng = np.random.default_rng(11)
    mu_true = normalize_quat(np.array([0.9, 0.2, -0.3, 0.1]))
    samples = _sample_component(mu_true, 0.0025 * np.eye(3), 200, rng)
    assert quat_dist(mu_true, karcher_mean(samples)) < 0.01


def test_rgmm_recovery():
    rng = np.random.default_rng(12)
    true_means = np.vstack(
        [
            np.array([1.0, 0.0, 0.0, 0.0]),
            quat_exp(np.array([1.3, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
            quat_exp(np.array([0.0, 1.2, 0.1]), np.array([1.0, 0.0, 0.0, 0.0])),
        ]
    )
    weights = np.array([0.4, 0.3, 0.3])
    counts = (weights * 500).astype(int)
    samples = np.vstack([_sample_component(mu, 0.0015 * np.eye(3), n, rng)
                         for mu, n in zip(true_means, counts)])
    model = RiemannianGMM(3, max_iter=50, random_state=13).fit(samples)
    distances = np.array([[quat_dist(model.means_[i], true_means[j]) for j in range(3)] for i in range(3)])
    row_ind, col_ind = linear_sum_assignment(distances)
    assert distances[row_ind, col_ind].max() < 0.05
    assert np.allclose(model.weights_[row_ind], weights[col_ind], atol=0.02)
    assert np.all(np.diff(model.log_likelihood_history_) >= -1e-6)


def test_vs_euclidean():
    rng = np.random.default_rng(14)
    means = np.vstack(
        [
            np.array([1.0, 0.0, 0.0, 0.0]),
            quat_exp(np.array([1.45, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
            quat_exp(np.array([0.0, 1.45, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
        ]
    )
    train = np.vstack([_sample_component(mu, 0.015 * np.eye(3), 180, rng) for mu in means])
    heldout = np.vstack([_sample_component(mu, 0.015 * np.eye(3), 60, rng) for mu in means])
    q_att = karcher_mean(train)
    rgmm = RiemannianGMM(3, max_iter=50, random_state=15).fit(train)
    baseline = _EuclideanGMM(3, max_iter=50, random_state=15).fit(np.vstack([quat_log(q, q_att) for q in train]))
    rgmm_ll = float(np.sum(rgmm.score_samples(heldout)))
    baseline_ll = float(np.sum(baseline.score_samples(np.vstack([quat_log(q, q_att) for q in heldout]))))
    print(f"RGMM log-likelihood: {rgmm_ll}")
    print(f"Single-chart GMM log-likelihood: {baseline_ll}")
    print(f"Improvement: {rgmm_ll - baseline_ll}")
    assert rgmm_ll >= baseline_ll


# ---- demo / viz ----


def _ellipsoid_points(covariance, mean_tangent, scale=1.0, n=18):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    sphere = np.array(
        [
            [np.cos(ui) * np.sin(vi), np.sin(ui) * np.sin(vi), np.cos(vi)]
            for ui in u
            for vi in v
        ]
    )
    vals, vecs = np.linalg.eigh(covariance)
    radii = np.sqrt(np.maximum(vals, 0.0)) * scale
    points = sphere @ np.diag(radii) @ vecs.T + mean_tangent
    return points.reshape(n, n, 3)


def demo_with_visualization(output_path="rgmm_demo.png"):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(20)
    means = np.vstack(
        [
            np.array([1.0, 0.0, 0.0, 0.0]),
            quat_exp(np.array([1.4, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
            quat_exp(np.array([0.0, 1.4, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
        ]
    )
    data = np.vstack([_sample_component(mu, 0.018 * np.eye(3), 160, rng) for mu in means])
    q_att = karcher_mean(data)
    x_chart = np.vstack([quat_log(q, q_att) for q in data])
    baseline = _EuclideanGMM(3, random_state=21).fit(x_chart)
    rgmm = RiemannianGMM(3, random_state=21).fit(data)

    labels_baseline = baseline.predict(x_chart)
    labels_rgmm = rgmm.predict(data)
    fig = plt.figure(figsize=(12, 5))
    titles = [
        "Single-chart Euclidean GMM in log_q_att",
        "Riemannian GMM (ellipsoids projected for display)",
    ]
    for idx, labels in enumerate([labels_baseline, labels_rgmm], start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        ax.scatter(x_chart[:, 0], x_chart[:, 1], x_chart[:, 2], c=labels, s=9, alpha=0.55)
        if idx == 1:
            mean_points = baseline.means_
            covariances = baseline.covariances_
        else:
            mean_points = np.vstack([quat_log(mu, q_att) for mu in rgmm.means_])
            covariances = rgmm.covariances_
        for mean, cov in zip(mean_points, covariances):
            ax.scatter(mean[0], mean[1], mean[2], c="black", s=70, marker="x")
            pts = _ellipsoid_points(cov, mean, scale=1.0)
            ax.plot_wireframe(pts[:, :, 0], pts[:, :, 1], pts[:, :, 2], color="black", alpha=0.18)
        ax.set_title(titles[idx - 1])
        ax.set_xlabel("log x")
        ax.set_ylabel("log y")
        ax.set_zlabel("log z")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


# ---- main ----


if __name__ == "__main__":
    tests = [test_roundtrip, test_karcher, test_rgmm_recovery, test_vs_euclidean]
    for test in tests:
        test()
        print(f"✓ {test.__name__}")
    saved = demo_with_visualization()
    print(f"✓ Demo figure saved: {saved}")
