from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import seaborn as sns

log_dir = "Fedours_18-02-2025_23:10:49/server/FedAdServer"  # Path to the event file
ea = event_accumulator.EventAccumulator(log_dir, size_guidance={"scalars": 0})
ea.Reload()

# Get available scalar tags
print(ea.Tags()["scalars"])

# Extract and plot a specific scalar
scalar_name = "val_loss"  # e.g., "loss"
events = ea.Scalars(scalar_name)
steps = [e.step for e in events]
values = [e.value for e in events]

values_mod = [val + 0.04 for val in values]

sns.set(style="darkgrid")  # Set Seaborn style

plt.figure(figsize=(8, 5))  # Optional: Adjust figure size
sns.lineplot(x=steps, y=values, label=scalar_name)
sns.lineplot(x=steps, y=values_mod, label=scalar_name)

plt.xlabel("Step")
plt.ylabel("Value")
plt.legend()
plt.show()
