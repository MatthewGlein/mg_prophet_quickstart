# Python
import pandas as pd
from fbprophet import Prophet

df = pd.read_csv('wp_log_peyton_manning.csv')
df.head()

# Python
m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
future.tail()

# Python
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)

fig1.savefig("Graphics/fig1.png")
fig2.savefig("Graphics/fig2.png")
