# ISA-PM-SLS-2021-01-09-IS02PT-GRP-SmartTrader

Large volumes of financial instruments are traded daily on exchanges and a growing majority of these are traded using automated trading algorithms. An automated trading strategy that maximizes profit is highly desirable for investors, mutual funds and hedge funds.
These trading algos can be broadly categorized into:

 - **Passive / Fundamental Trading** which is primarily a long-term strategy based on market and security fundamentals or passively benchmark index trading
 
 - **Active / Price Action Trading** which is based on buying and selling securities based on short-term movements to profit from the price movements and relies on historical prices (open, high, low, and close) to make trading decisions

This project will focus on the **Price Action Trading** which is driven on the characteristics of a security’s price movements. Since it ignores the more subjective fundamental factors and focuses solely on recent and past price data, the Price Action Trading Strategy is more conducive to Machine Learning. 

There are essentially two contrasting approaches to developing price trading strategies:

1. **Model based**:  This approach attempts to create a mathematical model of the market thru representative variables, such as mean price and corelations, to create a simplified representation of the true complex market model. Some examples of this are trend-following, mean reversion and arbitrage strategies.

2.	**Model free**: Here there is no attempt to model the market, rather it looks at price patterns and attempt to fit an algorithm to it. There is no attempt to produce a causal analysis or explanation – just an attempt to identify patterns that will repeat in the future. Some examples will include technical analysis, charting and candle patterns. 

This project will focus on the **Model free** approach and leverage Reinforcement Learning (RL) to develop an automated trading agent (Smart Trader) driven solely on historical data and derived technical features.
