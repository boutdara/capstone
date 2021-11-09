use group2

-- View all columns in census table
-- SELECT * FROM dbo.census_data



-- Create Keys for tables by first making relevant columns NOT NULL
ALTER TABLE
  dbo.ticker_data
ALTER COLUMN
  Ticker_ID int NOT NULL


ALTER TABLE dbo.ticker_data
ADD PRIMARY KEY (Ticker_ID)


ALTER TABLE
  dbo.stock_data
ALTER COLUMN
  Ticker_ID int NOT NULL

ALTER TABLE dbo.stock_data
ADD FOREIGN KEY (Ticker_ID) REFERENCES dbo.ticker_data(Ticker_ID)


ALTER TABLE
  dbo.stock_data
ALTER COLUMN
  Stock_ID int NOT NULL


ALTER TABLE dbo.stock_data
ADD PRIMARY KEY (Stock_ID)


ALTER TABLE
  dbo.SP_data
ALTER COLUMN
  SP_ID int NOT NULL


ALTER TABLE dbo.SP_data
ADD PRIMARY KEY (SP_ID)



ALTER TABLE
  dbo.census_data
ALTER COLUMN
  NAICS2017 int NOT NULL


ALTER TABLE dbo.census_data
ADD PRIMARY KEY (NAICS2017)


select * FROM dbo.ticker_data

select * FROM dbo.stock_data


-- Create View that joins tables to be used in project

-- DROP VIEW [stock_table]

-- CREATE VIEW [stock_table] AS
-- SELECT s.Date, t.Ticker_ID, Ticker, [Open], High, Low, [Close],
-- Volume, Perc, SP_Open, SP_High, SP_Low, SP_Close, SP_perc
-- FROM dbo.stock_data s
-- LEFT JOIN dbo.SP_data p ON s.Date = p.Date
-- LEFT JOIN dbo.ticker_data t ON t.Ticker_ID = s.Ticker_ID


SELECT * FROM [stock_table]