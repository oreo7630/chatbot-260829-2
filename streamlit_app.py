"""오늘의 뉴스 챗봇 - Streamlit 버전.

Google News RSS(한국)에서 오늘 발행된 뉴스를 가져와 사이드바에 보여주고,
OpenAI API로 뉴스에 대해 대화할 수 있는 챗봇입니다.
"""
import os
from datetime import datetime, timedelta, timezone

import feedparser
import streamlit as st
from openai import OpenAI

GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko"
KST = timezone(timedelta(hours=9))
MODEL = "gpt-4o-mini"


def get_api_key():
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    return os.environ.get("OPENAI_API_KEY")


def _parse_entry(entry):
    published = getattr(entry, "published_parsed", None)
    if not published:
        return None
    published_dt = datetime(*published[:6], tzinfo=timezone.utc).astimezone(KST)
    source = entry.get("source", {})
    source_title = source.get("title", "") if isinstance(source, dict) else ""
    return {
        "title": entry.title,
        "link": entry.link,
        "source": source_title,
        "published_dt": published_dt,
        "published": published_dt.strftime("%m/%d %H:%M"),
    }


@st.cache_data(ttl=600)
def fetch_today_news(limit=15):
    """오늘(KST) 발행된 뉴스를 최신순으로 가져옵니다.
    오늘자 기사가 부족하면 최신 기사로 채웁니다.
    """
    feed = feedparser.parse(GOOGLE_NEWS_RSS_URL)
    today = datetime.now(KST).date()
    parsed = [a for a in (_parse_entry(e) for e in feed.entries) if a]
    parsed.sort(key=lambda a: a["published_dt"], reverse=True)
    todays = [a for a in parsed if a["published_dt"].date() == today]
    articles = todays if todays else parsed
    return articles[:limit]


def build_system_prompt(articles):
    if not articles:
        news_block = "오늘 수집된 뉴스가 없습니다."
    else:
        news_block = "\n".join(
            f"{i + 1}. [{a['source']}] {a['title']} - {a['link']}"
            for i, a in enumerate(articles)
        )
    return (
        "당신은 '오늘의 뉴스' 챗봇입니다. 아래는 오늘 수집된 최신 뉴스 목록입니다.\n"
        "사용자의 질문에는 이 목록을 근거로 답하고, 목록에 없는 내용이면 모른다고 솔직히 말하세요.\n"
        "요약을 요청하면 간결하게 정리하고, 관련 기사 링크를 함께 안내하세요.\n\n"
        f"[오늘의 뉴스 목록]\n{news_block}"
    )


st.set_page_config(page_title="오늘의 뉴스 챗봇", page_icon="📰", layout="wide")

api_key = get_api_key()
articles = fetch_today_news()

with st.sidebar:
    st.header("오늘의 뉴스")
    if not articles:
        st.write("오늘 수집된 뉴스가 없습니다.")
    for a in articles:
        st.markdown(f"**[{a['title']}]({a['link']})**")
        st.caption(f"{a['source']} · {a['published']}")
        st.divider()

st.title("📰 오늘의 뉴스 챗봇")
st.caption("오늘의 뉴스에 대해 무엇이든 물어보세요. 예) \"오늘 뉴스 요약해줘\", \"경제 관련 소식 있어?\"")

if not api_key:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. Streamlit secrets 또는 환경변수를 확인하세요.")
    st.stop()

client = OpenAI(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            api_messages = [{"role": "system", "content": build_system_prompt(articles)}]
            api_messages.extend(st.session_state.messages)
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=api_messages,
                    temperature=0.4,
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"오류가 발생했습니다: {e}"
            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
