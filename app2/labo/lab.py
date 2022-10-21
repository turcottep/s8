import matplotlib.pyplot as plt
import numpy as np


def main():
    # TODO L2.E1.1

    all_means = []
    all_covs = []

    plt.figure()

    for i in range(10):

        sample = np.random.default_rng().multivariate_normal(
            mean=[3, -1], cov=[[1, 0], [0, 1]], size=20
        )

        plt.plot(sample[:, 0], sample[:, 1], "o")
        # get non-biased sample mean and covariance
        mean = np.mean(sample, axis=0)
        cov = np.cov(sample, rowvar=False)

        all_means.append(mean)
        all_covs.append(cov)

    print(f"all_means: {all_means}")
    print(f"all_means: {all_means}")

    all_means_mean = np.mean(all_means, axis=0)
    all_means_std = np.std(all_means, axis=0)
    all_covs_mean = np.mean(all_covs, axis=0)
    all_covs_std = np.std(all_covs, axis=0)

    print(f"all_means_mean: {all_means_mean}")
    print(f"all_means_std: {all_means_std}")
    print(f"all_covs_mean: {all_covs_mean}")
    print(f"all_covs_std: {all_covs_std}")

    plt.show()

    # TODO L2.E1.2
    # TODO L2.E1.3
    # TODO L2.E2.1
    # TODO L2.E2.2
    # TODO L2.E3
    print("Done")


if __name__ == "__main__":
    main()
