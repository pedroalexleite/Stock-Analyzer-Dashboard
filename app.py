#PACKAGES
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import praw
import re
import random
from textblob import TextBlob
from datetime import datetime, timedelta
from collections import Counter
import json
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

#CONFIGURATIONS
st.set_page_config(
    page_title="Stock Analyzer Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

#SENTIMENT AUX
REDDIT_CONFIG = {
    'client_id': st.secrets["reddit"]["client_id"],
    'client_secret': st.secrets["reddit"]["client_secret"],
    'user_agent': st.secrets["reddit"]["user_agent"]
}

@st.cache_data(ttl=3600)
def scrape_reddit(ticker, subreddit_name, time_filter="year"):
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CONFIG['client_id'],
            client_secret=REDDIT_CONFIG['client_secret'],
            user_agent=REDDIT_CONFIG['user_agent']
        )
        subreddit = reddit.subreddit(subreddit_name)
        posts = []

        if len(ticker) == 1:
            query = f"${ticker}"
        else:
            query = f"{ticker} OR ${ticker}"

        for submission in subreddit.search(
            query=query,
            sort="new",
            limit=100,
            syntax="lucene",
            time_filter=time_filter
        ):
            text = submission.title + " " + submission.selftext
            if text.strip():
                posts.append({
                    "text": text,
                    "created": datetime.fromtimestamp(submission.created_utc)
                })

        return posts
    except Exception as e:
        st.error(f"Reddit API error: {e}")
        return []

def clean_post(post):
    post = post.lower()
    post = re.sub(r'http\S+|www\S+', '', post)
    post = post.replace("\\", "")
    post = re.sub(r'\n\d*', ' ', post)
    post = re.sub(r'[\(\)\[\]\"\'-]', '', post)
    post = re.sub(r'[^\w\s]', '', post)

    try:
        words = word_tokenize(post)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.lower() not in stop_words and len(word) > 2]

        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        post = ' '.join(words)
    except:
        pass

    post = re.sub(r'\s+', ' ', post).strip()
    return post

def extract_sentiment(post):
    blob = TextBlob(post)
    return blob.sentiment.polarity

def analyze_sentiment_by_period(posts, ticker):
    now = datetime.now()

    periods = {
        '1W': now - timedelta(days=7),
        '1M': now - timedelta(days=30),
        '1Y': now - timedelta(days=365),
        'MAX': datetime.min
    }

    sentiment_by_period = {}

    for period_name, start_date in periods.items():
        period_posts = [p for p in posts if p['created'] >= start_date]

        if period_posts:
            sentiments = []
            for post in period_posts:
                cleaned = clean_post(post['text'])
                if cleaned:
                    score = extract_sentiment(cleaned)
                    sentiments.append(score)

            if sentiments:
                avg_sentiment = (sum(sentiments) / len(sentiments) + 1) * 50
                sentiment_by_period[period_name] = round(avg_sentiment, 1)
            else:
                sentiment_by_period[period_name] = 50
        else:
            sentiment_by_period[period_name] = 50

    return sentiment_by_period

def extract_keywords(posts, top_n=12):
    all_words = []
    stop_words = set(stopwords.words('english'))

    for post in posts:
        cleaned = clean_post(post['text'])
        if cleaned:
            words = cleaned.split()
            all_words.extend(words)

    if not all_words:
        return []

    word_counts = Counter(all_words)

    filtered_counts = {word: count for word, count in word_counts.items()
                       if word.lower() not in stop_words and len(word) > 3}

    top_words = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return top_words

#TITLE AND SEARCH
st.markdown("""
    <style>
    .stMarkdown h1 a,
    .stMarkdown h2 a,
    .stMarkdown h3 a,
    .stMarkdown h4 a,
    .stMarkdown h5 a,
    .stMarkdown h6 a {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Stock Analyzer Dashboard</h1>", unsafe_allow_html=True)
st.markdown("""
<style>
    /* Reduce space between title and search */
    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        max-width: 100%;
    }
    h1 {
        margin-bottom: 0rem !important;
        margin-top: 0rem !important;
        padding-top: 0.5rem !important;
    }

    input[type="text"] {
        text-align: center;
        font-size: 20px !important;
    }

    /* Remove extra spacing */
    .element-container {
        margin-bottom: 0 !important;
    }

    /* Remove all default Streamlit spacing */
    .stMarkdown {
        margin-bottom: 0 !important;
    }

    div[data-testid="column"] {
        padding-top: 0 !important;
    }
</style>
""", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1.5, 1, 1.5])
with col2:
    ticker = st.text_input("", placeholder="Enter stock ticker (e.g. AAPL, MSFT, TSLA, etc)", label_visibility="collapsed")

#AFTER SEARCH
if ticker:
    try:
        st.markdown('<div style="margin-top: -50px;"></div>', unsafe_allow_html=True)

        prices = yf.download(ticker, period="max", interval="1d", progress=False)["Close"]
        if prices.empty:
            st.error(f"No data found for '{ticker.upper()}'. Check the ticker symbol.")
        else:
            #AUX
            stock = yf.Ticker(ticker)
            info = stock.info

            hist_data = yf.download(ticker, period="max", interval="1d", progress=False)

            def human_format(num):
                if num is None:
                    return "N/A"
                num = float(num)
                if num >= 1_000_000_000_000:
                    return f"${round(num / 1_000_000_000_000, 2)}T"
                elif num >= 1_000_000_000:
                    return f"${round(num / 1_000_000_000, 2)}B"
                elif num >= 1_000_000:
                    return f"${round(num / 1_000_000, 2)}M"
                elif num >= 1_000:
                    return f"${round(num / 1_000, 2)}K"
                else:
                    return str(round(num, 2))

            def human_format_no_dollar(num):
                if num is None:
                    return "N/A"
                num = float(num)
                if num >= 1_000_000_000_000:
                    return f"{round(num / 1_000_000_000_000, 2)}T"
                elif num >= 1_000_000_000:
                    return f"{round(num / 1_000_000_000, 2)}B"
                elif num >= 1_000_000:
                    return f"{round(num / 1_000_000, 2)}M"
                elif num >= 1_000:
                    return f"{round(num / 1_000, 2)}K"
                else:
                    return str(round(num, 2))

            def calculate_sma(data, window=20):
                return data.rolling(window=window).mean()

            def calculate_ema(data, window=20):
                return data.ewm(span=window, adjust=False).mean()

            def calculate_macd(data):
                ema12 = data.ewm(span=12, adjust=False).mean()
                ema26 = data.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()
                return macd, signal

            def calculate_rsi(data, window=14):
                delta = data.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi

            def calculate_roc(data, window=12):
                roc = ((data - data.shift(window)) / data.shift(window)) * 100
                return roc

            def calculate_obv(close, volume):
                obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
                return obv

            ceo_name = "N/A"
            try:
                officers = info.get('companyOfficers', [])
                if officers and len(officers) > 0:
                    ceo_name = officers[0].get('name', 'N/A')
            except:
                pass

            left_col, right_col = st.columns([1, 1], gap="large")

            #LEFT HALF
            with left_col:
                col_chart, col_desc = st.columns([2, 1], gap="large")

                #CHART
                with col_chart:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=prices.index,
                        y=prices.values.flatten(),
                        mode="lines",
                        name=ticker.upper(),
                        line=dict(color="#ff9500")
                    ))
                    fig.update_xaxes(
                        rangeslider_visible=True,
                        rangeselector=dict(
                            buttons=list([
                                dict(label="MAX", step="all"),
                                dict(count=1, label="1Y", step="year", stepmode="backward"),
                                dict(count=1, label="1M", step="month", stepmode="backward"),
                                dict(count=7, label="1W", step="day", stepmode="backward")
                            ])
                        )
                    )
                    fig.update_layout(
                        xaxis_title=None,
                        yaxis_title=None,
                        height=500,
                        margin=dict(t=50, b=0, l=0, r=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                #DESCRIPTION
                with col_desc:
                    st.markdown('<div style="margin-top: 35px;"></div>', unsafe_allow_html=True)
                    st.markdown(f"**Name:** {info.get('shortName', 'N/A')}")
                    st.markdown(f"**Exchange:** {info.get('exchange', 'N/A')}")
                    st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
                    st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
                    st.markdown(f"**Country:** {info.get('country', 'N/A')}")
                    st.markdown(f"**City:** {info.get('city', 'N/A')}, {info.get('state', 'N/A')}")
                    st.markdown(f"**CEO:** {ceo_name}")
                    st.markdown(f"**Employees:** {human_format_no_dollar(info.get('fullTimeEmployees'))}")
                    st.markdown(f"**Price:** ${round(info.get('currentPrice', 0), 2)}")
                    st.markdown(f"**Target:** ${round(info.get('targetMeanPrice', 0), 2)}")
                    st.markdown(f"**Cap:** {human_format(info.get('marketCap'))}")
                    st.markdown(f"**Value:** {human_format(info.get('enterpriseValue'))}")

                st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
                fund_risk_col1, fund_risk_col2 = st.columns(2)

                #FUNDAMENTAL
                with fund_risk_col1:
                    st.markdown("<h4 style='text-align: center;'>Fundamental</h4>", unsafe_allow_html=True)

                    per = info.get('trailingPE', 'N/A')
                    pbr = info.get('priceToBook', 'N/A')
                    roa = info.get('returnOnAssets', 'N/A')
                    npm = info.get('profitMargins', 'N/A')
                    icr = info.get('currentRatio', 'N/A')
                    fcf = info.get('freeCashflow', 'N/A')
                    rgr = info.get('revenueGrowth', 'N/A')
                    egr = info.get('earningsGrowth', 'N/A')

                    def format_metric(value):
                        if value == 'N/A' or value is None:
                            return 'N/A'
                        try:
                            if isinstance(value, (int, float)):
                                return f"{round(value, 2)}"
                            return str(value)
                        except:
                            return 'N/A'

                    fundamental_table = f"""
                    <style>
                        .fundamental-table {{
                            width: 100%;
                            border-collapse: collapse;
                            margin-top: 20px;
                            font-size: 14px;
                            font-family: sans-serif;
                        }}
                        .fundamental-table th {{
                            background-color: #1d1d1d;
                            color: white;
                            padding: 12px;
                            text-align: center;
                            border: 2px solid #2C2E35;
                            font-weight: bold;
                        }}
                        .fundamental-table td {{
                            background-color: #1d1d1d;
                            color: white;
                            padding: 12px;
                            text-align: center;
                            border: 2px solid #2C2E35;
                        }}
                    </style>
                    <table class="fundamental-table">
                        <tr>
                            <th>Evaluation</th>
                            <th>Profitability</th>
                            <th>Health</th>
                            <th>Growth</th>
                        </tr>
                        <tr>
                            <td>PER: {format_metric(per)}</td>
                            <td>ROA: {format_metric(roa)}</td>
                            <td>ICR: {format_metric(icr)}</td>
                            <td>RGR: {format_metric(rgr)}</td>
                        </tr>
                        <tr>
                            <td>PBR: {format_metric(pbr)}</td>
                            <td>NPM: {format_metric(npm)}</td>
                            <td>FCF: {human_format(fcf) if fcf != 'N/A' else 'N/A'}</td>
                            <td>EGR: {format_metric(egr)}</td>
                        </tr>
                    </table>
                    """
                    st.markdown(fundamental_table, unsafe_allow_html=True)

                    #RISK
                    with fund_risk_col2:
                        st.markdown("<h4 style='text-align: center; margin-bottom: 0px;'>Risk</h4>", unsafe_allow_html=True)

                        col_left, col_center, col_right = st.columns([1.75, 2, 1.75])
                        with col_center:
                            risk_type = st.segmented_control(
                                "Select Risk Type",
                                ["Market", "Downside"],
                                default="Market",
                                label_visibility="collapsed",
                                key="risk_selector"
                            )

                        try:
                            risk_data = yf.download(ticker, period="1y", progress=False)

                            if not risk_data.empty:
                                if isinstance(risk_data.columns, pd.MultiIndex):
                                    risk_data.columns = risk_data.columns.get_level_values(0)

                                daily_returns = risk_data['Close'].pct_change().dropna()

                                volatility_value = daily_returns.std() * np.sqrt(252) * 100

                                beta_raw = info.get('beta')
                                if beta_raw is None or pd.isna(beta_raw):
                                    spy_data = yf.download("SPY", period="1y", progress=False)
                                    if isinstance(spy_data.columns, pd.MultiIndex):
                                        spy_data.columns = spy_data.columns.get_level_values(0)
                                    spy_returns = spy_data['Close'].pct_change().dropna()

                                    aligned_returns = pd.concat([daily_returns, spy_returns], axis=1, join='inner')
                                    aligned_returns.columns = ['stock', 'market']

                                    if len(aligned_returns) > 0:
                                        covariance = aligned_returns.cov().iloc[0, 1]
                                        market_variance = aligned_returns['market'].var()
                                        beta_raw = covariance / market_variance if market_variance != 0 else 1.0
                                    else:
                                        beta_raw = 1.0

                                beta_normalized = min(max((beta_raw / 2.0) * 100, 0), 100)

                                risk_free_rate = 0.04 / 252
                                excess_returns = daily_returns - risk_free_rate
                                downside_returns = excess_returns[excess_returns < 0]
                                downside_std = downside_returns.std() * np.sqrt(252)

                                if downside_std != 0:
                                    sortino_raw = (daily_returns.mean() * 252 - 0.04) / downside_std
                                else:
                                    sortino_raw = 0

                                sortino_normalized = min(max((sortino_raw / 3.0) * 100, 0), 100)

                                cumulative = (1 + daily_returns).cumprod()
                                running_max = cumulative.expanding().max()
                                drawdown = (cumulative - running_max) / running_max
                                max_drawdown = abs(drawdown.min() * 100)

                            else:
                                volatility_value = 24
                                beta_raw = 1.2
                                beta_normalized = 60
                                sortino_raw = 1.8
                                sortino_normalized = 60
                                max_drawdown = 35

                        except Exception as e:
                            volatility_value = 24
                            beta_raw = 1.2
                            beta_normalized = 60
                            sortino_raw = 1.8
                            sortino_normalized = 60
                            max_drawdown = 35

                        if risk_type == "Market":
                            fig_risk = go.Figure()

                            fig_risk.add_trace(go.Indicator(
                                mode="gauge",
                                value=volatility_value,
                                domain={'x': [0, 0.48], 'y': [0.3, 0.95]},
                                gauge={
                                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white", 'tickfont': {'size': 9}},
                                    'bar': {'color': "rgba(0,0,0,0)", 'thickness': 0},
                                    'bgcolor': "rgba(0,0,0,0)",
                                    'borderwidth': 2,
                                    'bordercolor': "#2C2E36",
                                    'threshold': {
                                        'line': {'color': "orange", 'width': 4},
                                        'thickness': 0.75,
                                        'value': volatility_value
                                    }
                                }
                            ))

                            fig_risk.add_trace(go.Indicator(
                                mode="gauge",
                                value=beta_normalized,
                                domain={'x': [0.52, 1], 'y': [0.3, 0.95]},
                                gauge={
                                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white", 'tickfont': {'size': 9}},
                                    'bar': {'color': "rgba(0,0,0,0)", 'thickness': 0},
                                    'bgcolor': "rgba(0,0,0,0)",
                                    'borderwidth': 2,
                                    'bordercolor': "#2C2E36",
                                    'threshold': {
                                        'line': {'color': "orange", 'width': 4},
                                        'thickness': 0.75,
                                        'value': beta_normalized
                                    }
                                }
                            ))

                            fig_risk.add_annotation(
                                text="Volatility",
                                x=0.24, y=0.25,
                                xref="paper", yref="paper",
                                showarrow=False,
                                font=dict(size=15, color="white"),
                                xanchor='center'
                            )

                            fig_risk.add_annotation(
                                text="Beta",
                                x=0.76, y=0.25,
                                xref="paper", yref="paper",
                                showarrow=False,
                                font=dict(size=15, color="white"),
                                xanchor='center'
                            )

                            fig_risk.update_layout(
                                height=200,
                                margin=dict(l=0, r=0, t=10, b=40),
                                paper_bgcolor='rgba(0,0,0,0)'
                            )
                            st.plotly_chart(fig_risk, use_container_width=True)

                        else:
                            fig_risk = go.Figure()

                            fig_risk.add_trace(go.Indicator(
                                mode="gauge",
                                value=sortino_normalized,
                                domain={'x': [0, 0.48], 'y': [0.3, 0.95]},
                                gauge={
                                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white", 'tickfont': {'size': 9}},
                                    'bar': {'color': "rgba(0,0,0,0)", 'thickness': 0},
                                    'bgcolor': "rgba(0,0,0,0)",
                                    'borderwidth': 2,
                                    'bordercolor': "#2C2E36",
                                    'threshold': {
                                        'line': {'color': "orange", 'width': 4},
                                        'thickness': 0.75,
                                        'value': sortino_normalized
                                    }
                                }
                            ))

                            fig_risk.add_trace(go.Indicator(
                                mode="gauge",
                                value=max_drawdown,
                                domain={'x': [0.52, 1], 'y': [0.3, 0.95]},
                                gauge={
                                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white", 'tickfont': {'size': 9}},
                                    'bar': {'color': "rgba(0,0,0,0)", 'thickness': 0},
                                    'bgcolor': "rgba(0,0,0,0)",
                                    'borderwidth': 2,
                                    'bordercolor': "#2C2E36",
                                    'threshold': {
                                        'line': {'color': "orange", 'width': 4},
                                        'thickness': 0.75,
                                        'value': max_drawdown
                                    }
                                }
                            ))

                            fig_risk.add_annotation(
                                text="Sortino",
                                x=0.24, y=0.25,
                                xref="paper", yref="paper",
                                showarrow=False,
                                font=dict(size=15, color="white"),
                                xanchor='center'
                            )

                            fig_risk.add_annotation(
                                text="Max Drawdown",
                                x=0.76, y=0.25,
                                xref="paper", yref="paper",
                                showarrow=False,
                                font=dict(size=15, color="white"),
                                xanchor='center'
                            )

                            fig_risk.update_layout(
                                height=200,
                                margin=dict(l=0, r=0, t=10, b=40),
                                paper_bgcolor='rgba(0,0,0,0)'
                            )
                            st.plotly_chart(fig_risk, use_container_width=True)

            #RIGHT HALF
            with right_col:
                #SENTIMENTAL
                st.markdown("<h4 style='text-align: center; margin-bottom: 0px;'>Sentimental</h4>", unsafe_allow_html=True)

                st.markdown("""
                <style>
                div[data-testid="stSelectbox"] {
                    margin-top: -20px !important;
                }
                </style>
                """, unsafe_allow_html=True)

                col_left, col_center, col_right = st.columns([36, 15, 36])
                with col_center:
                    sentiment_source = st.selectbox(
                        "Select Sentiment Source",
                        ["r/wallstreetbets", "r/stocks", "r/StockMarket", "r/investing"],
                        label_visibility="collapsed",
                        key="sentiment_selector"
                    )
                subreddit_name = sentiment_source.replace("r/", "")

                with st.spinner(f"Analyzing sentiment from {sentiment_source}..."):
                    posts = scrape_reddit(ticker, subreddit_name, time_filter="year")

                    if posts:
                        sentiment_data = analyze_sentiment_by_period(posts, ticker)
                        keywords = extract_keywords(posts, top_n = 15)
                    else:
                        sentiment_data = {'1W': 50, '1M': 50, '1Y': 50, 'MAX': 50}
                        keywords = []
                        st.warning(f"No posts found for {ticker.upper()} in {sentiment_source}")

                sent_left, sent_right = st.columns([1, 1])

                #BAR CHART
                with sent_left:
                    periods = ['1W', '1M', '1Y', 'MAX']
                    sentiment_values = [sentiment_data.get(p, 50) for p in periods]

                    fig_sentiment = go.Figure()

                    fig_sentiment.add_trace(go.Bar(
                        x=sentiment_values,
                        y=periods,
                        orientation='h',
                        marker=dict(
                            color='rgba(0,0,0,0)',
                            line=dict(color='#ff9500', width=2)
                        ),
                        text=sentiment_values,
                        textposition='outside',
                        textfont=dict(size=18, color='white'),
                        width=0.4
                    ))

                    fig_sentiment.update_layout(
                        height=200,
                        margin=dict(l=50, r=50, t=10, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(
                            title=None,
                            range=[0, 100],
                            tickfont=dict(size=10, color='white'),
                            showgrid=False,
                            showticklabels=False
                        ),
                        yaxis=dict(
                            title=None,
                            tickfont=dict(size=13, color='white'),
                            showgrid=False
                        ),
                        showlegend=False
                    )

                    st.plotly_chart(fig_sentiment, use_container_width=True)

                #WORDCLOUD
                with sent_right:
                    if keywords:
                        words_data = keywords
                    else:
                        words_data = [
                            ("ANALYSIS", 20), ("GROWTH", 18), ("VALUE", 16), ("MARKET", 14),
                            ("BULLISH", 12), ("HOLD", 10), ("INVESTMENT", 9), ("TREND", 8),
                            ("PERFORMANCE", 7), ("EARNINGS", 6), ("DIVIDEND", 5), ("RALLY", 4)
                        ]

                    random.seed(hash(sentiment_source + ticker) % 2**32)

                    fig_wordcloud = go.Figure()

                    for i, (word, weight) in enumerate(words_data):
                        x_pos = random.uniform(0.1, 0.9)
                        y_pos = random.uniform(0.1, 0.9)

                        max_weight = max([w[1] for w in words_data]) if words_data else 1
                        normalized_weight = weight / max_weight if max_weight > 0 else 0.5
                        font_size = 10 + (normalized_weight * 25)

                        fig_wordcloud.add_annotation(
                            text=word.upper(),
                            x=x_pos,
                            y=y_pos,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(
                                size=font_size,
                                color="#ff9500",
                                family="sans-serif"
                            ),
                            xanchor='center',
                            yanchor='middle'
                        )

                    fig_wordcloud.update_layout(
                        height=200,
                        margin=dict(l=0, r=0, t=10, b=10),
                        paper_bgcolor='#1d1d1d',
                        plot_bgcolor='#1d1d1d',
                        xaxis=dict(
                            showgrid=False,
                            showticklabels=False,
                            zeroline=False,
                            range=[0, 1]
                        ),
                        yaxis=dict(
                            showgrid=False,
                            showticklabels=False,
                            zeroline=False,
                            range=[0, 1]
                        )
                    )

                    st.plotly_chart(fig_wordcloud, use_container_width=True)

                #TECHNICAL
                st.markdown("<h4 style='text-align: center; margin-top: 20px;'>Technical</h4>", unsafe_allow_html=True)

                st.markdown("""
                <style>
                div[data-testid="stDateInput"] input {
                    font-size: 16px !important;
                }
                div[data-testid="stDateInput"] label {
                    font-size: 16px !important;
                    font-weight: 700 !important;
                    text-align: center !important;
                    display: block !important;
                }
                div[data-testid="stDateInput"] {
                    margin-bottom: 10px !important;
                }
                div[data-testid="stDateInput"] label p {
                    font-size: 16px !important;
                    font-weight: 700 !important;
                }
                </style>
                """, unsafe_allow_html=True)

                tech_controls_col, tech_chart_col = st.columns([1, 3])

                with tech_controls_col:
                    from datetime import datetime, timedelta
                    today = datetime.now()
                    one_year_ago = today - timedelta(days=365)

                    tech_start_date = st.date_input("Start", value=one_year_ago, key="tech_start", label_visibility="visible")
                    tech_end_date = st.date_input("End", value=today, key="tech_end", label_visibility="visible")

                    st.markdown("<p style='text-align: center; font-size: 16px; font-weight: bold; margin-top: 5px; margin-bottom: 10px;'>Indicators</p>", unsafe_allow_html=True)

                    if 'tech_indicators_list' not in st.session_state:
                        st.session_state.tech_indicators_list = ["SMA (20)"]

                    name_mapping = {
                        "Bolli Band (10)": "BB (10)",
                        "Bolli Band (20)": "BB (20)",
                        "MACD Hist": "MACD H",
                        "Stoch Osci": "SO",
                        "Will %R": "W%R",
                        "Par SAR": "P SAR",
                        "Ichi Cloud": "IC"
                    }
                    st.session_state.tech_indicators_list = [
                        name_mapping.get(ind, ind) for ind in st.session_state.tech_indicators_list
                    ]

                    temp_selection = st.multiselect(
                        "Select Indicators:",
                        [
                            "SMA (20)", "SMA (50)", "SMA (200)",
                            "EMA (12)", "EMA (26)", "EMA (50)",
                            "BB (10)", "BB (20)", "VWAP",
                            "RSI (9)", "RSI (14)",
                            "MACD", "MACD H",
                            "SO", "W%R",
                            "ATR (14)", "ADX (14)",
                            "P SAR", "IC"
                        ],
                        default=st.session_state.tech_indicators_list,
                        key="tech_indicators",
                        label_visibility="collapsed"
                    )

                    if len(temp_selection) <= 4:
                        tech_indicators = temp_selection
                        st.session_state.tech_indicators_list = temp_selection
                    else:
                        tech_indicators = st.session_state.tech_indicators_list
                        st.rerun()

                with tech_chart_col:
                    try:
                        tech_data = yf.download(ticker, start=tech_start_date, end=tech_end_date, progress=False)

                        if not tech_data.empty:
                            if isinstance(tech_data.columns, pd.MultiIndex):
                                tech_data.columns = tech_data.columns.get_level_values(0)

                            fig_tech = go.Figure(data=[
                                go.Scatter(
                                    x=tech_data.index,
                                    y=tech_data['Close'].values.flatten() if hasattr(tech_data['Close'], 'values') else tech_data['Close'],
                                    mode='lines',
                                    name="Price",
                                    line=dict(color='#ff9500')
                                )
                            ])

                            added_indicators = set()

                            for indicator in tech_indicators:
                                if indicator.startswith("SMA"):
                                    period = int(indicator.split("(")[1].split(")")[0])
                                    sma = tech_data['Close'].rolling(window=period).mean()
                                    show_legend = "SMA" not in added_indicators
                                    fig_tech.add_trace(go.Scatter(
                                        x=tech_data.index,
                                        y=sma,
                                        mode='lines',
                                        name='SMA',
                                        legendgroup='SMA',
                                        showlegend=show_legend,
                                        line=dict(color='#EA3323', width=2)
                                    ))
                                    added_indicators.add("SMA")

                                elif indicator.startswith("EMA"):
                                    period = int(indicator.split("(")[1].split(")")[0])
                                    ema = tech_data['Close'].ewm(span=period).mean()
                                    show_legend = "EMA" not in added_indicators
                                    fig_tech.add_trace(go.Scatter(
                                        x=tech_data.index,
                                        y=ema,
                                        mode='lines',
                                        name='EMA',
                                        legendgroup='EMA',
                                        showlegend=show_legend,
                                        line=dict(color='#E9337E', width=2)
                                    ))
                                    added_indicators.add("EMA")

                                elif indicator.startswith("BB"):
                                    period = int(indicator.split("(")[1].split(")")[0])
                                    sma = tech_data['Close'].rolling(window=period).mean()
                                    std = tech_data['Close'].rolling(window=period).std()
                                    bb_upper = sma + 2 * std
                                    bb_lower = sma - 2 * std
                                    show_legend = "BB" not in added_indicators
                                    fig_tech.add_trace(go.Scatter(
                                        x=tech_data.index,
                                        y=bb_upper,
                                        mode='lines',
                                        name='BB',
                                        legendgroup='BB',
                                        showlegend=show_legend,
                                        line=dict(color='#EA33F7', width=2)
                                    ))
                                    fig_tech.add_trace(go.Scatter(
                                        x=tech_data.index,
                                        y=bb_lower,
                                        mode='lines',
                                        name='BB',
                                        legendgroup='BB',
                                        showlegend=False,
                                        line=dict(color='#EA33F7', width=2)
                                    ))
                                    added_indicators.add("BB")

                                elif indicator == "VWAP":
                                    vwap = (tech_data['Close'] * tech_data['Volume']).cumsum() / tech_data['Volume'].cumsum()
                                    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=vwap, mode='lines', name='VWAP', line=dict(color='#7515F5', width=2)))

                                elif indicator.startswith("RSI"):
                                    period = int(indicator.split("(")[1].split(")")[0])
                                    delta = tech_data['Close'].diff()
                                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                                    rs = gain / loss
                                    rsi = 100 - (100 / (1 + rs))
                                    price_min = tech_data['Close'].min()
                                    price_max = tech_data['Close'].max()
                                    rsi_scaled = price_min + (rsi / 100) * (price_max - price_min)
                                    show_legend = "RSI" not in added_indicators
                                    fig_tech.add_trace(go.Scatter(
                                        x=tech_data.index,
                                        y=rsi_scaled,
                                        mode='lines',
                                        name='RSI',
                                        legendgroup='RSI',
                                        showlegend=show_legend,
                                        line=dict(color='#1401F5', width=2)
                                    ))
                                    added_indicators.add("RSI")

                                elif indicator == "MACD":
                                    ema12 = tech_data['Close'].ewm(span=12, adjust=False).mean()
                                    ema26 = tech_data['Close'].ewm(span=26, adjust=False).mean()
                                    macd = ema12 - ema26
                                    signal = macd.ewm(span=9, adjust=False).mean()
                                    price_min = tech_data['Close'].min()
                                    price_max = tech_data['Close'].max()
                                    macd_min = macd.min()
                                    macd_max = macd.max()
                                    if macd_max != macd_min:
                                        macd_scaled = price_min + ((macd - macd_min) / (macd_max - macd_min)) * (price_max - price_min) * 0.3
                                        signal_scaled = price_min + ((signal - macd_min) / (macd_max - macd_min)) * (price_max - price_min) * 0.3
                                        fig_tech.add_trace(go.Scatter(x=tech_data.index, y=macd_scaled, mode='lines', name='MACD', line=dict(color='#377EF6', width=2)))
                                        fig_tech.add_trace(go.Scatter(x=tech_data.index, y=signal_scaled, mode='lines', name='MACD', legendgroup='MACD', showlegend=False, line=dict(color='#377EF6', width=2)))

                                elif indicator == "MACD H":
                                    ema12 = tech_data['Close'].ewm(span=12, adjust=False).mean()
                                    ema26 = tech_data['Close'].ewm(span=26, adjust=False).mean()
                                    macd = ema12 - ema26
                                    signal = macd.ewm(span=9, adjust=False).mean()
                                    histogram = macd - signal
                                    price_min = tech_data['Close'].min()
                                    price_max = tech_data['Close'].max()
                                    hist_min = histogram.min()
                                    hist_max = histogram.max()
                                    if hist_max != hist_min:
                                        hist_scaled = price_min + ((histogram - hist_min) / (hist_max - hist_min)) * (price_max - price_min) * 0.2
                                        fig_tech.add_trace(go.Scatter(x=tech_data.index, y=hist_scaled, mode='lines', name='MACD H', fill='tozeroy', line=dict(color='#377EF6', width=2)))

                                elif indicator == "SO":
                                    low_14 = tech_data['Low'].rolling(window=14).min()
                                    high_14 = tech_data['High'].rolling(window=14).max()
                                    stoch_k = 100 * ((tech_data['Close'] - low_14) / (high_14 - low_14))
                                    stoch_d = stoch_k.rolling(window=3).mean()
                                    price_min = tech_data['Close'].min()
                                    price_max = tech_data['Close'].max()
                                    stoch_k_scaled = price_min + (stoch_k / 100) * (price_max - price_min)
                                    stoch_d_scaled = price_min + (stoch_d / 100) * (price_max - price_min)
                                    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=stoch_k_scaled, mode='lines', name='SO', line=dict(color='#75FBFD', width=2)))
                                    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=stoch_d_scaled, mode='lines', name='SO', legendgroup='SO', showlegend=False, line=dict(color='#75FBFD', width=2)))

                                elif indicator == "W%R":
                                    high_14 = tech_data['High'].rolling(window=14).max()
                                    low_14 = tech_data['Low'].rolling(window=14).min()
                                    williams_r = -100 * ((high_14 - tech_data['Close']) / (high_14 - low_14))
                                    price_min = tech_data['Close'].min()
                                    price_max = tech_data['Close'].max()
                                    williams_r_scaled = price_min + ((williams_r + 100) / 100) * (price_max - price_min)
                                    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=williams_r_scaled, mode='lines', name='W%R', line=dict(color='#75FC8E', width=2)))

                                elif indicator == "ATR (14)":
                                    high_low = tech_data['High'] - tech_data['Low']
                                    high_close = np.abs(tech_data['High'] - tech_data['Close'].shift())
                                    low_close = np.abs(tech_data['Low'] - tech_data['Close'].shift())
                                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                                    atr = tr.rolling(window=14).mean()
                                    price_min = tech_data['Close'].min()
                                    atr_scaled = price_min + atr
                                    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=atr_scaled, mode='lines', name='ATR (14)', line=dict(color='#75FB4C', width=2)))

                                elif indicator == "ADX (14)":
                                    high_diff = tech_data['High'].diff()
                                    low_diff = -tech_data['Low'].diff()
                                    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
                                    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
                                    high_low = tech_data['High'] - tech_data['Low']
                                    high_close = np.abs(tech_data['High'] - tech_data['Close'].shift())
                                    low_close = np.abs(tech_data['Low'] - tech_data['Close'].shift())
                                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                                    atr = tr.rolling(window=14).mean()
                                    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
                                    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
                                    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
                                    adx = dx.rolling(window=14).mean()
                                    price_min = tech_data['Close'].min()
                                    price_max = tech_data['Close'].max()
                                    adx_scaled = price_min + (adx / 100) * (price_max - price_min)
                                    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=adx_scaled, mode='lines', name='ADX (14)', line=dict(color='#A1FC4E', width=2)))

                                elif indicator == "P SAR":
                                    sar = tech_data['Close'].copy()
                                    af = 0.02
                                    uptrend = True
                                    ep = tech_data['High'].iloc[0]

                                    for i in range(1, len(tech_data)):
                                        if uptrend:
                                            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
                                            if tech_data['Low'].iloc[i] < sar.iloc[i]:
                                                uptrend = False
                                                sar.iloc[i] = ep
                                                ep = tech_data['Low'].iloc[i]
                                                af = 0.02
                                            else:
                                                if tech_data['High'].iloc[i] > ep:
                                                    ep = tech_data['High'].iloc[i]
                                                    af = min(af + 0.02, 0.2)
                                        else:
                                            sar.iloc[i] = sar.iloc[i-1] - af * (sar.iloc[i-1] - ep)
                                            if tech_data['High'].iloc[i] > sar.iloc[i]:
                                                uptrend = True
                                                sar.iloc[i] = ep
                                                ep = tech_data['High'].iloc[i]
                                                af = 0.02
                                            else:
                                                if tech_data['Low'].iloc[i] < ep:
                                                    ep = tech_data['Low'].iloc[i]
                                                    af = min(af + 0.02, 0.2)

                                    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=sar, mode='markers', name='P SAR', marker=dict(color='#FEFF54', size=3)))

                                elif indicator == "IC":
                                    nine_period_high = tech_data['High'].rolling(window=9).max()
                                    nine_period_low = tech_data['Low'].rolling(window=9).min()
                                    tenkan_sen = (nine_period_high + nine_period_low) / 2

                                    period26_high = tech_data['High'].rolling(window=26).max()
                                    period26_low = tech_data['Low'].rolling(window=26).min()
                                    kijun_sen = (period26_high + period26_low) / 2

                                    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

                                    period52_high = tech_data['High'].rolling(window=52).max()
                                    period52_low = tech_data['Low'].rolling(window=52).min()
                                    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

                                    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=tenkan_sen, mode='lines', name='IC', line=dict(color='#E0AA58', width=2)))
                                    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=kijun_sen, mode='lines', name='IC', legendgroup='IC', showlegend=False, line=dict(color='#E0AA58', width=2)))
                                    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=senkou_span_a, mode='lines', name='IC', legendgroup='IC', showlegend=False, line=dict(color='#E0AA58', width=2)))
                                    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=senkou_span_b, mode='lines', name='IC', legendgroup='IC', showlegend=False, line=dict(color='#E0AA58', width=2), fill='tonexty'))

                            fig_tech.update_layout(
                                xaxis_rangeslider_visible=False,
                                height=375,
                                margin=dict(l=0, r=0, t=0, b=0),
                                xaxis_title=None,
                                yaxis_title=None
                            )

                            st.plotly_chart(fig_tech, use_container_width=True)
                        else:
                            st.warning("No technical data available for the selected date range.")

                    except Exception as e:
                        st.error(f"Error loading technical data: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            #BOTTOM HALF
            analysis_col, chat_col = st.columns([1, 1], gap="large")

            GROQ_API_KEY = st.secrets["groq"]["api_key"]

            def calculate_technical_indicators(tech_data, tech_start_date, tech_end_date):
                try:
                    if tech_data.empty:
                        return {}

                    if isinstance(tech_data.columns, pd.MultiIndex):
                        tech_data.columns = tech_data.columns.get_level_values(0)

                    technical_metrics = {}

                    sma_20 = tech_data['Close'].rolling(window=20).mean().iloc[-1]
                    sma_50 = tech_data['Close'].rolling(window=50).mean().iloc[-1]
                    sma_200 = tech_data['Close'].rolling(window=200).mean().iloc[-1]
                    current_price = tech_data['Close'].iloc[-1]

                    technical_metrics['sma_20'] = round(sma_20, 2) if not pd.isna(sma_20) else 'N/A'
                    technical_metrics['sma_50'] = round(sma_50, 2) if not pd.isna(sma_50) else 'N/A'
                    technical_metrics['sma_200'] = round(sma_200, 2) if not pd.isna(sma_200) else 'N/A'
                    technical_metrics['price_vs_sma20'] = round(((current_price - sma_20) / sma_20) * 100, 2) if not pd.isna(sma_20) else 'N/A'
                    technical_metrics['price_vs_sma50'] = round(((current_price - sma_50) / sma_50) * 100, 2) if not pd.isna(sma_50) else 'N/A'

                    ema_12 = tech_data['Close'].ewm(span=12, adjust=False).mean().iloc[-1]
                    ema_26 = tech_data['Close'].ewm(span=26, adjust=False).mean().iloc[-1]
                    technical_metrics['ema_12'] = round(ema_12, 2) if not pd.isna(ema_12) else 'N/A'
                    technical_metrics['ema_26'] = round(ema_26, 2) if not pd.isna(ema_26) else 'N/A'

                    delta = tech_data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    technical_metrics['rsi_14'] = round(rsi.iloc[-1], 2) if not pd.isna(rsi.iloc[-1]) else 'N/A'

                    macd = ema_12 - ema_26
                    signal = pd.Series(macd).ewm(span=9, adjust=False).mean()
                    technical_metrics['macd'] = round(macd, 2) if not pd.isna(macd) else 'N/A'
                    technical_metrics['macd_signal'] = round(signal.iloc[-1] if isinstance(signal, pd.Series) else signal, 2) if not pd.isna(signal.iloc[-1] if isinstance(signal, pd.Series) else signal) else 'N/A'

                    sma_20_bb = tech_data['Close'].rolling(window=20).mean()
                    std_20 = tech_data['Close'].rolling(window=20).std()
                    bb_upper = sma_20_bb + 2 * std_20
                    bb_lower = sma_20_bb - 2 * std_20
                    technical_metrics['bb_upper'] = round(bb_upper.iloc[-1], 2) if not pd.isna(bb_upper.iloc[-1]) else 'N/A'
                    technical_metrics['bb_lower'] = round(bb_lower.iloc[-1], 2) if not pd.isna(bb_lower.iloc[-1]) else 'N/A'
                    technical_metrics['bb_position'] = round(((current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])) * 100, 2) if not pd.isna(bb_upper.iloc[-1]) and not pd.isna(bb_lower.iloc[-1]) else 'N/A'

                    high_diff = tech_data['High'].diff()
                    low_diff = -tech_data['Low'].diff()
                    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
                    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
                    high_low = tech_data['High'] - tech_data['Low']
                    high_close = np.abs(tech_data['High'] - tech_data['Close'].shift())
                    low_close = np.abs(tech_data['Low'] - tech_data['Close'].shift())
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    atr = tr.rolling(window=14).mean()
                    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
                    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
                    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
                    adx = dx.rolling(window=14).mean()
                    technical_metrics['adx_14'] = round(adx.iloc[-1], 2) if not pd.isna(adx.iloc[-1]) else 'N/A'

                    technical_metrics['atr_14'] = round(atr.iloc[-1], 2) if not pd.isna(atr.iloc[-1]) else 'N/A'

                    low_14 = tech_data['Low'].rolling(window=14).min()
                    high_14 = tech_data['High'].rolling(window=14).max()
                    stoch_k = 100 * ((tech_data['Close'] - low_14) / (high_14 - low_14))
                    technical_metrics['stochastic'] = round(stoch_k.iloc[-1], 2) if not pd.isna(stoch_k.iloc[-1]) else 'N/A'

                    williams_r = -100 * ((high_14 - tech_data['Close']) / (high_14 - low_14))
                    technical_metrics['williams_r'] = round(williams_r.iloc[-1], 2) if not pd.isna(williams_r.iloc[-1]) else 'N/A'

                    vwap = (tech_data['Close'] * tech_data['Volume']).cumsum() / tech_data['Volume'].cumsum()
                    technical_metrics['vwap'] = round(vwap.iloc[-1], 2) if not pd.isna(vwap.iloc[-1]) else 'N/A'

                    return technical_metrics

                except Exception as e:
                    print(f"Error calculating technical indicators: {e}")
                    return {}

            @st.cache_data(ttl=86400)
            def generate_ai_analysis(ticker_symbol, stock_data, sentiment_data, risk_metrics, technical_metrics):
                try:
                    tech_indicators_str = "\n".join([f"    - {key.replace('_', ' ').title()}: {value}" for key, value in technical_metrics.items()])
                    analysis_prompt = f"""Analyze the stock {ticker_symbol} based on the following data:

            FUNDAMENTAL DATA:
            - P/E Ratio: {stock_data.get('per', 'N/A')}
            - Price to Book: {stock_data.get('pbr', 'N/A')}
            - Return on Assets: {stock_data.get('roa', 'N/A')}
            - Net Profit Margin: {stock_data.get('npm', 'N/A')}
            - Current Ratio: {stock_data.get('icr', 'N/A')}
            - Free Cash Flow: {stock_data.get('fcf', 'N/A')}
            - Revenue Growth: {stock_data.get('rgr', 'N/A')}
            - Earnings Growth: {stock_data.get('egr', 'N/A')}
            - Current Price: ${stock_data.get('price', 'N/A')}
            - Target Price: ${stock_data.get('target', 'N/A')}
            - Market Cap: {stock_data.get('marketCap', 'N/A')}
            - Sector: {stock_data.get('sector', 'N/A')}
            - Industry: {stock_data.get('industry', 'N/A')}

            RISK METRICS:
            - Volatility: {risk_metrics.get('volatility', 'N/A')}%
            - Beta: {risk_metrics.get('beta', 'N/A')}
            - Sortino Ratio: {risk_metrics.get('sortino', 'N/A')}
            - Max Drawdown: {risk_metrics.get('max_drawdown', 'N/A')}%

            TECHNICAL INDICATORS:
            {tech_indicators_str}

            SENTIMENT DATA (Reddit scores 0-100, where 50 is neutral):
            - 1 Week: {sentiment_data.get('1W', 'N/A')}
            - 1 Month: {sentiment_data.get('1M', 'N/A')}
            - 1 Year: {sentiment_data.get('1Y', 'N/A')}

            IMPORTANT INSTRUCTIONS:
            1. Provide a rating from 0-100 that reflects the OVERALL investment potential based on ALL factors above
            2. Consider how the different metrics interact - don't just average them
            3. Weight factors appropriately: fundamentals and technicals should be weighted heavily, sentiment moderately
            4. Each text analysis should be exactly 120-160 words - not shorter, not longer
            5. Write in plain text ONLY - no markdown formatting, no asterisks, no bold, no italic
            6. Be specific and reference the actual numbers provided
            7. Use actual percentages when appropriate and exactly 2 decimal points in all cases
            8. Make the analysis professional, balanced, and actionable

            Provide a JSON response ONLY (no markdown, no explanation) with this exact structure:
            {{
                "rating": <number 0-100 based on comprehensive analysis>,
                "summary_table": {{
                    "sentiment": "<Bullish/Bearish/Neutral>",
                    "fundamental": "<Strong/Weak/Average>",
                    "short_term": "<Risky/Moderate/Favorable>",
                    "action": "<Buy/Hold/Sell>",
                    "sentimental": "<Positive/Negative/Neutral>",
                    "medium_term": "<Risky/Moderate/Favorable>",
                    "value": "<Overvalued/Fair/Undervalued>",
                    "technical": "<Strong/Weak/Neutral>",
                    "long_term": "<Risky/Moderate/Favorable>"
                }},
                "detailed_analysis": {{
                    "risk": "<Plain text paragraph 120-160 words analyzing volatility, beta, sortino, and max drawdown>",
                    "fundamental": "<Plain text paragraph 120-160 words analyzing PE, PBR, ROA, NPM, cash flow, and growth metrics>",
                    "sentimental": "<Plain text paragraph 120-160 words analyzing Reddit sentiment across time periods>",
                    "technical": "<Plain text paragraph 120-160 words analyzing SMA, EMA, RSI, MACD, Bollinger Bands, ADX, and other technical indicators>",
                    "overall": "<Plain text paragraph 120-160 words providing comprehensive summary and investment recommendation>"
                }}
            }}"""

                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {GROQ_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "llama-3.3-70b-versatile",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a professional financial analyst. Always respond with valid JSON only, no markdown formatting. Write all text in plain format without any markdown symbols like asterisks, underscores, or bold/italic formatting. Each analysis section must be between 120-160 words."
                                },
                                {
                                    "role": "user",
                                    "content": analysis_prompt
                                }
                            ],
                            "temperature": 0.0,
                            "seed": 42,
                            "max_tokens": 4000
                        }
                    )

                    if response.status_code == 200:
                        data = response.json()
                        content_text = data['choices'][0]['message']['content']

                        content_text = content_text.strip()
                        if content_text.startswith('```json'):
                            content_text = content_text[7:]
                        if content_text.startswith('```'):
                            content_text = content_text[3:]
                        if content_text.endswith('```'):
                            content_text = content_text[:-3]

                        analysis_result = json.loads(content_text.strip())

                        for key in analysis_result.get('detailed_analysis', {}).keys():
                            text = analysis_result['detailed_analysis'][key]
                            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
                            text = re.sub(r'\*(.+?)\*', r'\1', text)
                            text = re.sub(r'_(.+?)_', r'\1', text)
                            text = text.replace('**', '').replace('*', '').replace('__', '').replace('_', '')
                            analysis_result['detailed_analysis'][key] = text

                        return analysis_result
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                        return None

                except json.JSONDecodeError as e:
                    st.error(f"Error parsing AI response: {e}")
                    return None
                except Exception as e:
                    st.error(f"Error generating AI analysis: {e}")
                    return None

            def get_chat_response(context, user_question, chat_history):
                try:
                    messages = [{"role": "system", "content": context}]

                    for msg in chat_history[-5:]:
                        messages.append({"role": msg["role"], "content": msg["content"]})

                    messages.append({"role": "user", "content": user_question})

                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {GROQ_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "llama-3.3-70b-versatile",
                            "messages": messages,
                            "temperature": 0.0,
                            "seed": 42,
                            "max_tokens": 500
                        }
                    )

                    if response.status_code == 200:
                        data = response.json()
                        return data['choices'][0]['message']['content']
                    else:
                        return f"Error: {response.status_code}"
                except Exception as e:
                    return f"Error: {e}"

            analysis_col, chat_col = st.columns([1, 1], gap="large")

            #ANALYSIS
            with analysis_col:
                st.markdown("<h4 style='text-align: center; margin-bottom: 20px; margin-top: -20px;'>Analysis</h4>", unsafe_allow_html=True)

                stock_data_for_ai = {
                    'per': per,
                    'pbr': pbr,
                    'roa': roa,
                    'npm': npm,
                    'icr': icr,
                    'fcf': fcf,
                    'rgr': rgr,
                    'egr': egr,
                    'price': info.get('currentPrice', 0),
                    'target': info.get('targetMeanPrice', 0),
                    'marketCap': info.get('marketCap'),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A')
                }

                risk_metrics_for_ai = {
                    'volatility': round(volatility_value, 2),
                    'beta': round(beta_raw, 2),
                    'sortino': round(sortino_raw, 2),
                    'max_drawdown': round(max_drawdown, 2)
                }

                tech_data_for_analysis = yf.download(ticker, start=tech_start_date, end=tech_end_date, progress=False)
                technical_metrics_for_ai = calculate_technical_indicators(tech_data_for_analysis, tech_start_date, tech_end_date)

                with st.spinner("Generating AI analysis..."):
                    ai_analysis = generate_ai_analysis(ticker, stock_data_for_ai, sentiment_data, risk_metrics_for_ai, technical_metrics_for_ai)

                if ai_analysis:
                    rating_col, table_col = st.columns([1, 2])

                    #RATING
                    with rating_col:
                        rating_value = ai_analysis['rating']

                        fig_rating = go.Figure()

                        fig_rating.add_trace(go.Indicator(
                            mode="gauge+number",
                            value=rating_value,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            number={'suffix': "", 'font': {'size': 36, 'color': 'white', 'family': 'sans-serif'}, 'valueformat': '.0f'},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': "white", 'visible': False},
                                'bar': {'color': "#ff9500", 'thickness': 0.7},
                                'bgcolor': "rgba(0,0,0,0)",
                                'borderwidth': 0,
                                'shape': "angular",
                                'steps': [
                                    {'range': [0, 100], 'color': '#2C2E35', 'thickness': 0.7}
                                ]
                            }
                        ))

                        fig_rating.update_layout(
                            height=150,
                            margin=dict(l=20, r=20, t=10, b=5),
                            paper_bgcolor='rgba(0,0,0,0)',
                            font={'color': 'white', 'weight': 'bold'}
                        )

                        st.plotly_chart(fig_rating, use_container_width=True)

                    #TABLE
                    with table_col:
                        summary = ai_analysis['summary_table']
                        analysis_table = f"""
                        <style>
                            .analysis-table {{
                                width: 100%;
                                border-collapse: collapse;
                                margin-top: 15px;
                                margin-bottom: 5px;
                                font-size: 13px;
                                font-family: sans-serif;
                                table-layout: fixed;
                            }}
                            .analysis-table td {{
                                background-color: #1d1d1d;
                                color: white;
                                padding: 10px;
                                text-align: left;
                                border: 2px solid #2C2E35;
                                vertical-align: middle;
                                width: 33.33%;
                            }}
                        </style>
                        <table class="analysis-table">
                            <tr>
                                <td><strong>Sentiment:</strong> {summary['sentiment']}</td>
                                <td><strong>Fundamental:</strong> {summary['fundamental']}</td>
                                <td><strong>Short-Term:</strong> {summary['short_term']}</td>
                            </tr>
                            <tr>
                                <td><strong>Action:</strong> {summary['action']}</td>
                                <td><strong>Sentimental:</strong> {summary['sentimental']}</td>
                                <td><strong>Medium-Term:</strong> {summary['medium_term']}</td>
                            </tr>
                            <tr>
                                <td><strong>Value:</strong> {summary['value']}</td>
                                <td><strong>Technical:</strong> {summary['technical']}</td>
                                <td><strong>Long-Term:</strong> {summary['long_term']}</td>
                            </tr>
                        </table>
                        """
                        st.markdown(analysis_table, unsafe_allow_html=True)

                    st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)

                    #EXPANDABLES
                    st.markdown("""
                    <style>
                        div[data-testid="stExpander"] {
                            border: 1px solid #2C2E35 !important;
                            border-radius: 4px;
                        }
                        div[data-testid="stExpander"] details {
                            border: 1px solid #2C2E35 !important;
                        }
                    </style>
                    """, unsafe_allow_html=True)

                    detailed = ai_analysis['detailed_analysis']

                    with st.expander("**Risk**", expanded=False):
                        st.write(detailed['risk'])

                    with st.expander("**Fundamental**", expanded=False):
                        st.write(detailed['fundamental'])

                    with st.expander("**Sentimental**", expanded=False):
                        st.write(detailed['sentimental'])

                    with st.expander("**Technical**", expanded=False):
                        st.write(detailed['technical'])

                    with st.expander("**Overall**", expanded=False):
                        st.write(detailed['overall'])
                else:
                    st.warning("AI analysis unavailable. Please check your API configuration.")

            #CHAT
            with chat_col:
                st.markdown("<h4 style='text-align: center; margin-bottom: 20px; margin-top: -20px;'>Chat</h4>", unsafe_allow_html=True)

                if f'chat_history_{ticker}' not in st.session_state:
                    st.session_state[f'chat_history_{ticker}'] = []

                st.markdown("""
                <style>
                    .stChatFloatingInputContainer {
                        bottom: 0px;
                    }

                    /* Style the chat messages container */
                    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
                        max-height: 380px;
                        overflow-y: auto;
                    }
                </style>
                """, unsafe_allow_html=True)

                messages_container = st.container(height=395)

                with messages_container:
                    for message in st.session_state[f'chat_history_{ticker}']:
                        with st.chat_message(message['role'], avatar="👤" if message['role'] == 'user' else "🤖"):
                            import html
                            escaped_content = html.escape(message['content'])
                            st.markdown(f"<div style='white-space: pre-wrap;'>{escaped_content}</div>", unsafe_allow_html=True)

                user_question = st.chat_input("Ask a question about this stock...")

                if user_question:
                    st.session_state[f'chat_history_{ticker}'].append({
                        'role': 'user',
                        'content': user_question
                    })

                    context = f"""You are a financial analyst assistant. Answer questions about {ticker} stock based on the following data:

                    COMPANY INFO:
                    - Name: {info.get('shortName', 'N/A')}
                    - Sector: {info.get('sector', 'N/A')}
                    - Industry: {info.get('industry', 'N/A')}
                    - Current Price: ${round(info.get('currentPrice', 0), 2)}
                    - Target Price: ${round(info.get('targetMeanPrice', 0), 2)}
                    - Market Cap: {human_format(info.get('marketCap'))}

                    FUNDAMENTAL METRICS:
                    - P/E Ratio: {format_metric(per)}
                    - Price to Book: {format_metric(pbr)}
                    - ROA: {format_metric(roa)}
                    - Net Profit Margin: {format_metric(npm)}
                    - Current Ratio: {format_metric(icr)}
                    - Free Cash Flow: {human_format(fcf) if fcf != 'N/A' else 'N/A'}
                    - Revenue Growth: {format_metric(rgr)}
                    - Earnings Growth: {format_metric(egr)}

                    RISK METRICS:
                    - Volatility: {round(volatility_value, 2)}%
                    - Beta: {round(beta_raw, 2)}
                    - Sortino Ratio: {round(sortino_raw, 2)}
                    - Max Drawdown: {round(max_drawdown, 2)}%

                    TECHNICAL INDICATORS:
                    {chr(10).join([f"- {key.replace('_', ' ').title()}: {value}" for key, value in technical_metrics_for_ai.items()])}

                    SENTIMENT (Reddit {sentiment_source}):
                    - 1 Week: {sentiment_data.get('1W', 'N/A')}/100
                    - 1 Month: {sentiment_data.get('1M', 'N/A')}/100
                    - 1 Year: {sentiment_data.get('1Y', 'N/A')}/100

                    Provide a clear, concise answer to the user's question. Be specific and reference actual data when relevant."""

                    with st.spinner("Thinking..."):
                        assistant_response = get_chat_response(
                            context,
                            user_question,
                            st.session_state[f'chat_history_{ticker}']
                        )

                        st.session_state[f'chat_history_{ticker}'].append({
                            'role': 'assistant',
                            'content': assistant_response
                        })

                        st.rerun()

    except Exception as e:
        st.error(f"Error fetching data: {e}")
