from partitions import partitions


class UnbalancedIIDMap(partitions.Partition):

    def generate(self, dataset, **kwargs):
        # def unbalanced_iid_partition(self, dataset, imbalance_factor=2):
        """
        Partition the dataset into multiple IID (Independent and Identically Distributed)
        subsets with different size.

        This function divides a dataset into a specified number of IID subsets (federated
        clients), where each subset has a different number of samples. The number of samples
        in each subset is determined by an imbalance factor, making the partition suitable
        for simulating imbalanced data scenarios in federated learning.

        Args:
            dataset (list): The dataset to partition. It should be a list of tuples where
                            each tuple represents a data sample and its corresponding label.
            imbalance_factor (float): The factor to determine the degree of imbalance
                                    among the subsets. A lower imbalance factor leads to more
                                    imbalanced partitions.

        Returns:
            dict: A dictionary where keys are client IDs (ranging from 0 to partitions_number-1) and
                    values are lists of indices corresponding to the samples assigned to each client.

        The function ensures that each class is represented in each subset but with varying
        proportions. The partitioning process involves iterating over each class, shuffling
        the indices of that class, and then splitting them according to the calculated subset
        sizes. The function does not print the class distribution in each subset.

        Example usage:
            federated_data = unbalanced_iid_partition(my_dataset, imbalance_factor=2)
            # This creates federated data subsets with varying number of samples based on
            # an imbalance factor of 2.
        """
        num_clients = self.partitions_number
        clients_data = {i: [] for i in range(num_clients)}

        # Get the labels from the dataset
        labels = np.array([dataset.targets[idx]
                          for idx in range(len(dataset))])
        label_counts = np.bincount(labels)

        min_label = label_counts.argmin()
        min_count = label_counts[min_label]

        # Set the initial_subset_size
        initial_subset_size = min_count // num_clients

        # Calculate the number of samples for each subset based on the imbalance factor
        subset_sizes = [initial_subset_size]
        for i in range(1, num_clients):
            subset_sizes.append(
                int(subset_sizes[i - 1] * ((imbalance_factor - 1) / imbalance_factor)))

        for label in range(self.num_classes):
            # Get the indices of the same label samples
            label_indices = np.where(labels == label)[0]
            np.random.seed(self.seed)
            np.random.shuffle(label_indices)

            # Split the data based on their labels
            start = 0
            for i in range(num_clients):
                end = start + subset_sizes[i]
                clients_data[i].extend(label_indices[start:end])
                start = end

        return clients_data
