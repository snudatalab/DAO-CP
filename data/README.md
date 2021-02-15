## Datasets
| No. | Name | Order | Dimensions | Description | Source |
|:--|:--|:--|:--|:--|:--|
| 1 | Synthetic data	| 4 | (1K, 10, 20, 30) 		| Synthetic tensor of various themes. consists of <br> (timestamp, custom mode1, custom mode2, custom mode3) | [Link](https://github.com/snudatalab/DAO-CP/)
| 2 | Sample video		| 4 | (205, 240, 320, 3) 	| Sample video on [YouTube](https://www.youtube.com/watch?v=EngW7tLk6R8). consists of <br>(frame, width, height, RGB colors) | [Link](https://www.sample-videos.com/)
| 3 | Stock price   	| 3 | (3K, 140, 5) 			| Korea stock price. consists of <br>(timestamp in date, stock, price type) | [Link](https://deeptrade.co/})
| 4 | Airport hall		| 3 | (200, 144, 176) 		| Airport hall video. consists of (frame, width, height) | [Link](https://github.com/hiroyuki-kasai/OLSTEC/)
| 5 | Korea air quality | 3 | (10K, 323, 6) 		| Korea air pollutants information. consists of <br>(timestamp in hour, location, atmospheric pollutants; measurement) | [Link](https://www.airkorea.or.kr/eng/)
<br>

1. Synthetic dataset is made of concatenated tensors which is summation of three tensors `T_main`, `T_theme`, and `T_noise`.<br>
Each tensor refers to 100x, 10x and 1x normal distributed randomized tensor. 
2. Sample video dataset is a series of animation frames whose pixel has its RGB value.
3. Stock price dataset involves 140 stocks data existing from Jan 2, 2008 to June 30, 2020 among Korea Stock Price Index 200 (KOSPI200). Features are sampled as adjusted opening/highest/lowest/closing price and trading ratio among total shares.
4. Airport hall dataset is made of a recorded video in an airport.
5. Air quality dataset is a set of daily measures in Seoul, Korea from Sep 1, 2018 to Sep 31, 2019.<br>
Measurements of pollutants are recorded varying to dates and locations.
