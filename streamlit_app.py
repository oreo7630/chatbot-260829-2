"""오늘의 뉴스 챗봇 - Streamlit 버전 (음성 입력/출력 지원).

Google News RSS(한국)에서 분야별 오늘 발행된 뉴스를 가져와 사이드바에 보여주고,
OpenAI API로 뉴스에 대해 대화할 수 있는 챗봇입니다.
텍스트뿐 아니라 음성으로 질문하고, 답변을 음성으로 들을 수도 있습니다.
"""

import hashlib
from datetime import datetime, timedelta, timezone
from io import BytesIO

import feedparser
import streamlit as st
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder

KST = timezone(timedelta(hours=9))
MODEL = "gpt-4o-mini"
STT_MODEL = "whisper-1"
TTS_MODEL = "tts-1"
TTS_VOICE = "alloy"

# --------------------------------------------------
# 분야별 Google News RSS 토픽
# --------------------------------------------------

CATEGORIES = {
    "정치": "NATION",
    "경제": "BUSINESS",
    "IT/과학": "TECHNOLOGY",
    "스포츠": "SPORTS",
    "연예": "ENTERTAINMENT",
}

TOP_N_PER_CATEGORY = 5


def _category_rss_url(topic_code: str) -> str:
    return (
        f"https://news.google.com/rss/headlines/section/topic/{topic_code}"
        "?hl=ko&gl=KR&ceid=KR:ko"
    )


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
        "published": published_dt.strftime("%H:%M"),
    }


@st.cache_data(ttl=600)
def fetch_category_news(topic_code: str, limit: int = TOP_N_PER_CATEGORY):
    """특정 분야(topic_code)의 오늘(KST) 뉴스를 최신순으로 최대 limit개 반환."""
    feed = feedparser.parse(_category_rss_url(topic_code))
    today = datetime.now(KST).date()

    parsed = [a for a in (_parse_entry(e) for e in feed.entries) if a]
    parsed.sort(key=lambda a: a["published_dt"], reverse=True)

    todays = [a for a in parsed if a["published_dt"].date() == today]
    articles = todays if todays else parsed
    return articles[:limit]


@st.cache_data(ttl=600)
def fetch_all_categories():
    """모든 분야의 top N 뉴스를 {분야명: [기사, ...]} 형태로 반환."""
    return {name: fetch_category_news(topic) for name, topic in CATEGORIES.items()}


def build_system_prompt(news_by_category):
    blocks = []
    for category, articles in news_by_category.items():
        if not articles:
            blocks.append(f"[{category}]\n(수집된 뉴스 없음)")
            continue
        lines = "\n".join(
            f"  {i + 1}. [{a['source']}] {a['title']} ({a['published']}) - {a['link']}"
            for i, a in enumerate(articles)
        )
        blocks.append(f"[{category}]\n{lines}")

    news_block = "\n\n".join(blocks)

    return (
        "당신은 '오늘의 뉴스' 챗봇입니다. 아래는 분야별로 정리된 오늘의 주요 뉴스 목록입니다.\n"
        "사용자의 질문에는 이 목록을 근거로 답하고, 목록에 없는 내용을 물으면 모른다고 솔직히 말하세요.\n"
        "특정 분야를 물으면 해당 분야만 정리해서 답하고, 요약을 요청하면 간결하게 정리한 뒤 "
        "관련 기사 링크를 함께 안내하세요.\n"
        "음성으로 읽힐 수 있으니 답변은 너무 길지 않게, 자연스러운 구어체로 정리하세요.\n\n"
        f"[분야별 오늘의 뉴스 목록]\n{news_block}"
    )


def transcribe_audio(client: OpenAI, audio_bytes: bytes) -> str:
    """녹음된 음성을 텍스트로 변환 (Whisper API)."""
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "recording.wav"
    transcript = client.audio.transcriptions.create(
        model=STT_MODEL,
        file=audio_file,
        language="ko",
    )
    return transcript.text


def synthesize_speech(client: OpenAI, text: str) -> bytes:
    """텍스트를 음성으로 변환 (TTS API)."""
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
    )
    return response.content


st.set_page_config(page_title="오늘의 뉴스 챗봇", page_icon="📰", layout="wide")

st.title("📰 오늘의 뉴스 챗봇")
st.caption(
    "정치·경제·IT/과학·스포츠·연예 분야별 오늘의 뉴스를 확인하고, "
    "텍스트나 음성으로 무엇이든 물어보세요."
)

# --------------------------------------------------
# API KEY (사용자가 직접 입력)
# --------------------------------------------------

api_key = st.text_input("OpenAI API Key", type="password")

if not api_key:
    st.info("OpenAI API Key를 입력해주세요.", icon="🔑")
    st.stop()

client = OpenAI(api_key=api_key)

news_by_category = fetch_all_categories()

# --------------------------------------------------
# 사이드바: 분야별 top 5 뉴스 + 음성 답변 설정
# --------------------------------------------------

with st.sidebar:
    st.header("오늘의 뉴스 (분야별 Top 5)")
    tabs = st.tabs(list(CATEGORIES.keys()))
    for tab, category in zip(tabs, CATEGORIES.keys()):
        with tab:
            articles = news_by_category.get(category, [])
            if not articles:
                st.write("수집된 뉴스가 없습니다.")
            for a in articles:
                st.markdown(f"**[{a['title']}]({a['link']})**")
                st.caption(f"{a['source']} · {a['published']}")
                st.divider()

    st.divider()
    st.header("🔊 음성 설정")
    voice_reply_enabled = st.checkbox("답변을 음성으로 듣기", value=False)

# --------------------------------------------------
# 대화 기록
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "audio" in msg:
            st.audio(msg["audio"], format="audio/mp3")

# --------------------------------------------------
# 입력: 텍스트 + 음성
# --------------------------------------------------

text_input = st.chat_input("메시지를 입력하세요")

st.markdown("**또는 음성으로 질문하기**")
audio_data = mic_recorder(
    start_prompt="🎤 녹음 시작",
    stop_prompt="⏹ 녹음 종료",
    just_once=False,
    key="recorder",
)

user_input = None

if text_input:
    user_input = text_input

elif audio_data and audio_data.get("bytes"):
    audio_hash = hashlib.md5(audio_data["bytes"]).hexdigest()
    if audio_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = audio_hash
        with st.spinner("음성을 텍스트로 변환 중..."):
            try:
                user_input = transcribe_audio(client, audio_data["bytes"])
            except Exception as e:
                st.error(f"음성 인식 중 오류가 발생했습니다: {e}")

# --------------------------------------------------
# 챗봇 응답
# --------------------------------------------------

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            api_messages = [{"role": "system", "content": build_system_prompt(news_by_category)}]
            api_messages.extend(
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-10:]
            )

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

            assistant_msg = {"role": "assistant", "content": reply}

            if voice_reply_enabled and not reply.startswith("오류가 발생했습니다"):
                with st.spinner("음성 생성 중..."):
                    try:
                        audio_bytes = synthesize_speech(client, reply)
                        st.audio(audio_bytes, format="audio/mp3")
                        assistant_msg["audio"] = audio_bytes
                    except Exception as e:
                        st.warning(f"음성 생성 중 오류가 발생했습니다: {e}")

    st.session_state.messages.append(assistant_msg)
