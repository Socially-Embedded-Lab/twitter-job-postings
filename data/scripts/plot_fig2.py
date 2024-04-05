import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(f'data/job_offer/dedup_counts.csv', parse_dates=['date'])
df = df.set_index('date')

plot_df = df['01.01.2015':]

label_size = 18
tick_size = 15

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.plot(plot_df['initial'], label='Job postings', color='red')
plt.plot(plot_df['total_tweets'], label='Deduplicated', color='green')
plt.xlabel('Date', fontsize=label_size)
plt.ylabel('# of Tweets', fontsize=label_size)
plt.xticks(fontsize=tick_size)  # Increase font size of x-axis tick labels
plt.yticks(fontsize=tick_size)  # Increase font size of y-axis tick labels
plt.tick_params(axis='both', which='major', length=6, width=1)  # Set tick sizes
plt.legend(fontsize=tick_size)

plt.savefig('data/job_offer/dedup_counts.pdf', format='pdf')

# Show the plot
plt.show()
