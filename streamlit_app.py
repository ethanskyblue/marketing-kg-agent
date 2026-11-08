"""
마케팅 지식그래프 기반 AI 에이전트 - Streamlit 웹앱

로컬 실행: streamlit run streamlit_app.py
배포: Streamlit Community Cloud(GitHub 저장소) 또는 Hugging Face Spaces에 이 파일 + requirements.txt 업로드

필요한 접속 정보(secrets)는 .streamlit/secrets.toml 또는 배포 플랫폼의 Secrets 설정에 등록합니다.
로그인 계정은 secrets가 아니라 Neo4j의 :User 노드(username, password_hash, status, is_admin)로 관리합니다.
[secrets.toml 예시]
ANTHROPIC_API_KEY = "sk-ant-..."
NEO4J_URI = "neo4j+s://xxxx.databases.neo4j.io"
NEO4J_USER = "xxxx"
NEO4J_PASSWORD = "..."
NEO4J_DATABASE = "xxxx"   # Aura credentials 파일의 NEO4J_DATABASE 값 (없으면 "neo4j"로 시도)
QDRANT_URL = "https://xxxx.cloud.qdrant.io"
QDRANT_API_KEY = "..."
CF_ACCOUNT_ID = "..."     # Cloudflare 계정 ID (질문 임베딩용 Workers AI)
CF_API_TOKEN = "..."      # Workers AI 권한의 API 토큰

회원가입 흐름:
- 사용자가 "회원가입 신청" 탭에서 아이디/비밀번호를 제출하면 status='pending'인 :User 노드가 생성됩니다.
- status가 'pending'인 계정은 로그인할 수 없습니다.
- is_admin=true인 계정으로 로그인하면 관리자 패널이 나타나 승인/거절할 수 있습니다.
- 최초 관리자 계정은 Colab의 manage_users.py로 미리 만들어둬야 합니다 (닭과 달걀 문제).

참고: Neo4j는 Bolt 프로토콜(포트 7687)이 아니라 Query API(HTTPS, 포트 443)로 접속합니다.
Hugging Face Spaces 등 일부 호스팅 환경은 80/443/8080 외 포트로 나가는 요청을 막기 때문에,
Bolt 드라이버(neo4j 패키지)를 쓰면 연결 시도가 응답 없이 멈춰버릴 수 있습니다.

다중 문서 지원: 노트북에서 문서마다 DOCUMENT_ID를 바꿔가며 처리하면 Qdrant에
"chunks_<DOCUMENT_ID>" 컬렉션이 각각 생성됩니다. 사이드바에서 골라 질문할 수 있습니다.
"""

import json
import math
import os
import time

import streamlit as st
import anthropic
import bcrypt
import requests
from qdrant_client import QdrantClient

COLLECTION_PREFIX = "chunks_"

st.set_page_config(page_title="마케팅 지식그래프 기반 AI 에이전트", page_icon="📄")

# 크롬 등 브라우저의 자동 번역이 한글 텍스트를 다시 오역하는 것을 막고,
# 홈 화면 아이콘 이름도 "마케팅 지식그래프 기반 AI 에이전트"로 표시되도록 페이지 메타 정보를 보정합니다.
# 참고: st.components.v1.html은 deprecated(2026-06-01 이후 제거 예정)이므로 st.iframe으로 대체합니다.
st.iframe(
    """
    <script>
    try {
        var p = window.parent.document;
        p.documentElement.lang = "ko";
        p.documentElement.setAttribute("translate", "no");

        var notranslate = p.createElement("meta");
        notranslate.name = "google";
        notranslate.content = "notranslate";
        p.head.appendChild(notranslate);

        p.title = "마케팅 지식그래프 기반 AI 에이전트";

        var link = p.querySelector('link[rel="manifest"]');
        var manifest = {
            name: "마케팅 지식그래프 기반 AI 에이전트",
            short_name: "마케팅AI에이전트",
            start_url: ".",
            display: "standalone"
        };
        var blob = new Blob([JSON.stringify(manifest)], {type: "application/json"});
        var manifestURL = URL.createObjectURL(blob);
        if (link) {
            link.setAttribute("href", manifestURL);
        } else {
            link = p.createElement("link");
            link.rel = "manifest";
            link.href = manifestURL;
            p.head.appendChild(link);
        }
    } catch (e) {}
    </script>
    """,
    height=1,  # st.iframe은 height=0을 허용하지 않음 (양의 정수 또는 'stretch'/'content' 필요) - 최소값으로 사실상 안 보이게 처리
)


def get_secret(key: str):
    """Streamlit Cloud(st.secrets)와 Hugging Face Spaces(환경변수) 양쪽에서 동작하도록 함"""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key)


# ---- Neo4j Query API - 로그인/회원가입에서도 필요하므로 무거운 리소스보다 먼저 정의 ----
@st.cache_resource
def get_neo4j_session():
    session = requests.Session()
    session.auth = (get_secret("NEO4J_USER"), get_secret("NEO4J_PASSWORD"))
    session.headers.update({"Content-Type": "application/json", "Accept": "application/json"})
    return session


def get_neo4j_endpoint():
    uri = get_secret("NEO4J_URI") or ""
    hostname = uri.replace("neo4j+s://", "").replace("neo4j://", "").strip("/")
    database = get_secret("NEO4J_DATABASE") or "neo4j"
    return f"https://{hostname}/db/{database}/query/v2"


def run_cypher(statement: str, parameters: dict = None):
    session = get_neo4j_session()
    resp = session.post(
        get_neo4j_endpoint(),
        json={"statement": statement, "parameters": parameters or {}},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    fields = data["fields"]
    return [dict(zip(fields, values)) for values in data["values"]]


# ---- 로그인 / 회원가입 ----
def find_user(username: str):
    rows = run_cypher(
        "MATCH (u:User {username: $username}) "
        "RETURN u.password_hash AS password_hash, u.status AS status, u.is_admin AS is_admin",
        {"username": username},
    )
    return rows[0] if rows else None


def create_signup_request(username: str, password: str):
    if not username or not password:
        return False, "아이디와 비밀번호를 모두 입력해주세요."
    if find_user(username):
        return False, "이미 등록되었거나 신청 중인 아이디입니다."

    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    run_cypher(
        """
        CREATE (u:User {
            username: $username,
            password_hash: $hash,
            status: 'pending',
            is_admin: false,
            requested_at: datetime()
        })
        """,
        {"username": username, "hash": password_hash},
    )
    return True, "회원가입 신청이 접수되었습니다. 관리자 승인 후 로그인할 수 있습니다."


def render_login_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #eef2fb 0%, #f8fafc 55%, #ffffff 100%);
        }
        .login-hero {
            text-align: center;
            padding-top: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .login-hero .icon {
            display: flex;
            justify-content: center;
            margin-bottom: 0.4rem;
        }
        .login-hero h1 {
            margin: 0.3rem 0 0 0;
            font-size: 1.5rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.3;
        }
        .login-hero .en {
            color: #64748b;
            margin-top: 0.1rem;
            font-size: 1rem;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 22px !important;
            padding: 0.4rem;
        }
        div[data-testid="stTextInput"] input {
            border-radius: 999px;
            padding: 0.65rem 1.1rem;
        }
        div[data-testid="stButton"] > button[kind="primary"] {
            background-color: #0f1b3d;
            border-color: #0f1b3d;
            border-radius: 999px;
            padding: 0.65rem 0;
            font-weight: 700;
            width: 100%;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background-color: #16264f;
            border-color: #16264f;
        }
        div[data-testid="stButton"] > button[kind="secondary"] {
            border: none;
            background: transparent;
            color: #475569;
            width: 100%;
        }
        div[data-testid="stButton"] > button[kind="secondary"]:hover {
            color: #0f1b3d;
            text-decoration: underline;
        }
        .block-container {
            max-width: 480px;
            margin: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_login_header():
    st.markdown(
        """
        <div class="login-hero">
            <div class="icon">
                <svg width="76" height="76" viewBox="0 0 76 76" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <linearGradient id="bgGrad" x1="0" y1="0" x2="1" y2="1">
                            <stop offset="0%" stop-color="#1e3a8a"/>
                            <stop offset="100%" stop-color="#0891b2"/>
                        </linearGradient>
                        <linearGradient id="badgeGrad" x1="0" y1="0" x2="1" y2="1">
                            <stop offset="0%" stop-color="#f97316"/>
                            <stop offset="100%" stop-color="#fb923c"/>
                        </linearGradient>
                        <filter id="softShadow" x="-30%" y="-30%" width="160%" height="160%">
                            <feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#0f172a" flood-opacity="0.18"/>
                        </filter>
                    </defs>
                    <rect x="3" y="3" width="70" height="70" rx="20" fill="url(#bgGrad)" filter="url(#softShadow)"/>
                    <rect x="21" y="16" width="27" height="36" rx="4" fill="white" opacity="0.97"/>
                    <rect x="26" y="24" width="17" height="2.4" rx="1.2" fill="#1e3a8a"/>
                    <rect x="26" y="30.5" width="17" height="2.4" rx="1.2" fill="#1e3a8a"/>
                    <rect x="26" y="37" width="10" height="2.4" rx="1.2" fill="#1e3a8a"/>
                    <circle cx="52" cy="50" r="14" fill="url(#badgeGrad)" stroke="white" stroke-width="2.5"/>
                    <text x="52" y="55" text-anchor="middle" font-size="12" font-weight="700" fill="white" font-family="Arial, sans-serif">AI</text>
                </svg>
            </div>
            <h1>마케팅 지식그래프 기반 AI 에이전트</h1>
            <div class="en">Marketing Knowledge Graph AI Agent</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def check_login() -> bool:
    if st.session_state.get("authenticated"):
        return True

    render_login_styles()
    render_login_header()

    view = st.session_state.get("_auth_view", "login")

    with st.container(border=True):
        if view == "login":
            st.markdown("**아이디**")
            username = st.text_input(
                "아이디", placeholder="👤 아이디를 입력하세요.",
                label_visibility="collapsed", key="login_username",
            )
            st.markdown("**비밀번호**")
            password = st.text_input(
                "비밀번호", placeholder="🔒 비밀번호를 입력하세요.", type="password",
                label_visibility="collapsed", key="login_password",
            )

            _l, _c, _r = st.columns([1, 1.6, 1])
            with _c:
                login_clicked = st.button("로그인", type="primary", key="login_submit", use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                forgot_clicked = st.button("비밀번호를 잊으셨나요?", type="secondary", key="forgot_pw")
            with col2:
                signup_link_clicked = st.button("회원가입", type="secondary", key="go_signup")

            if forgot_clicked:
                st.info("비밀번호 재설정은 관리자에게 문의해주세요.")

            if signup_link_clicked:
                st.session_state["_auth_view"] = "signup"
                st.rerun()

            if login_clicked:
                try:
                    user = find_user(username)
                except Exception as ex:
                    user = None
                    st.error(f"로그인 확인 중 오류가 발생했습니다: {ex}")
                else:
                    if not user or not user.get("password_hash"):
                        st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
                    elif not bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
                        st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
                    elif user.get("status") != "approved":
                        st.warning("아직 관리자 승인 대기 중인 계정입니다. 승인 후 다시 시도해주세요.")
                    else:
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = username
                        st.session_state["is_admin"] = bool(user.get("is_admin"))
                        st.rerun()

        else:  # 회원가입 화면
            st.markdown("**사용할 아이디**")
            new_username = st.text_input(
                "사용할 아이디", placeholder="👤 아이디를 입력하세요.",
                label_visibility="collapsed", key="signup_username",
            )
            st.markdown("**사용할 비밀번호**")
            new_password = st.text_input(
                "사용할 비밀번호", placeholder="🔒 비밀번호를 입력하세요.", type="password",
                label_visibility="collapsed", key="signup_password",
            )
            st.markdown("**비밀번호 확인**")
            new_password_confirm = st.text_input(
                "비밀번호 확인", placeholder="🔒 비밀번호를 다시 입력하세요.", type="password",
                label_visibility="collapsed", key="signup_password_confirm",
            )

            _l2, _c2, _r2 = st.columns([1, 1.6, 1])
            with _c2:
                signup_clicked = st.button("가입 신청", type="primary", key="signup_submit", use_container_width=True)
            back_clicked = st.button("← 로그인으로 돌아가기", type="secondary", key="back_to_login")

            if back_clicked:
                st.session_state["_auth_view"] = "login"
                st.rerun()

            if signup_clicked:
                if new_password != new_password_confirm:
                    st.error("비밀번호가 서로 일치하지 않습니다.")
                else:
                    try:
                        ok, message = create_signup_request(new_username, new_password)
                    except Exception as ex:
                        ok, message = False, f"신청 중 오류가 발생했습니다: {ex}"
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)

    return False


if not check_login():
    st.stop()

st.sidebar.caption(f"👤 {st.session_state.get('username', '')}" + (" (관리자)" if st.session_state.get("is_admin") else ""))
if st.sidebar.button("로그아웃"):
    st.session_state.clear()
    st.rerun()


# ---- 무거운 리소스는 로그인 이후에만 로드하고, 캐싱해서 한 번만 로드 ----
@st.cache_resource
def get_anthropic_client():
    return anthropic.Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))


@st.cache_resource
def get_qdrant_client():
    return QdrantClient(url=get_secret("QDRANT_URL"), api_key=get_secret("QDRANT_API_KEY"))


client = get_anthropic_client()
qdrant = get_qdrant_client()

# Neo4j 연결 확인 (실패해도 앱 자체는 계속 뜨도록 처리)
neo4j_ok = True
try:
    run_cypher("RETURN 1 AS ok")
except Exception as ex:
    neo4j_ok = False
    st.warning(f"Neo4j 연결에 실패했습니다 (그래프 확장 기능 없이 벡터 검색만 동작합니다): {ex}")


# ---- 관리자 패널: 회원가입 승인/거절 ----
if st.session_state.get("is_admin") and neo4j_ok:
    with st.expander("🛠️ 관리자 - 회원가입 승인 대기 목록", expanded=True):
        try:
            pending = run_cypher(
                "MATCH (u:User {status: 'pending'}) RETURN u.username AS username ORDER BY u.requested_at"
            )
        except Exception as ex:
            pending = []
            st.error(f"목록을 불러오지 못했습니다: {ex}")

        if not pending:
            st.caption("대기 중인 신청이 없습니다.")
        for row in pending:
            uname = row["username"]
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.write(uname)
            if col2.button("승인", key=f"approve_{uname}"):
                run_cypher("MATCH (u:User {username: $u}) SET u.status = 'approved'", {"u": uname})
                st.rerun()
            if col3.button("거절", key=f"reject_{uname}"):
                run_cypher("MATCH (u:User {username: $u}) DELETE u", {"u": uname})
                st.rerun()


# ---- 문서(컬렉션) 선택 ----
ALL_DOCS_LABEL = "🔎 전체 문서 통합 검색"


def list_document_collections():
    try:
        collections = qdrant.get_collections().collections
        return sorted(c.name for c in collections if c.name.startswith(COLLECTION_PREFIX))
    except Exception:
        return []


def collection_display_name(name: str) -> str:
    return name if name == ALL_DOCS_LABEL else name[len(COLLECTION_PREFIX):]


available_collections = list_document_collections()

with st.sidebar:
    st.markdown("## 📚 문서 선택")
    st.caption("질문할 문서를 골라주세요. 여러 문서를 한 번에 검색할 수도 있어요.")
    with st.container(border=True):
        if available_collections:
            options = [ALL_DOCS_LABEL] + available_collections
            COLLECTION_NAME = st.selectbox(
                "질문할 문서",
                options,
                format_func=collection_display_name,
                label_visibility="collapsed",
            )
        else:
            COLLECTION_NAME = st.text_input("컬렉션 이름 직접 입력", value="chunks_doc1")

        if COLLECTION_NAME == ALL_DOCS_LABEL:
            st.success(f"🔎 업로드된 문서 {len(available_collections)}개를 모두 검색합니다.")
        else:
            st.info(f"📄 현재 선택: **{collection_display_name(COLLECTION_NAME)}**")
    st.divider()

if st.session_state.get("_active_collection") != COLLECTION_NAME:
    st.session_state.messages = []
    st.session_state["_active_collection"] = COLLECTION_NAME


# ---- 질문 임베딩: Cloudflare Workers AI의 bge-m3 (앱 안에서 모델을 돌리지 않음) ----
# Colab의 로컬 bge-m3 벡터와 코사인 유사도 1.0000으로 일치함을 확인했습니다(CLS 풀링, 1024차원).
def embed_query(text: str) -> list:
    account_id = (get_secret("CF_ACCOUNT_ID") or "").strip()
    token = (get_secret("CF_API_TOKEN") or "").strip()
    if not account_id or not token:
        raise RuntimeError("CF_ACCOUNT_ID 또는 CF_API_TOKEN이 설정되지 않았습니다.")
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/baai/bge-m3"
    last = None
    for attempt in range(3):
        try:
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {token}"},
                json={"text": [text], "truncate_inputs": True},
                timeout=20,
            )
            r.raise_for_status()
            body = r.json()
            if not body.get("success", True):
                raise RuntimeError(body.get("errors"))
            vec = body["result"]["data"][0]
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            return [x / norm for x in vec]
        except Exception as ex:
            last = ex
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"임베딩 요청 실패: {last}")


# ---- 하이브리드 검색: Qdrant 벡터 검색 + (연결되면) Neo4j 관계 확장 ----
# COLLECTION_NAME이 ALL_DOCS_LABEL이면 모든 문서 컬렉션을 각각 검색한 뒤,
# 유사도 점수 기준으로 합쳐서 상위 top_k개만 남깁니다.
def hybrid_search(query: str, top_k: int = 5):
    query_vec = embed_query(query)

    if COLLECTION_NAME == ALL_DOCS_LABEL:
        target_collections = available_collections
    else:
        target_collections = [COLLECTION_NAME]

    scored_hits = []  # (score, hit, collection_name) 튜플 목록
    for coll in target_collections:
        try:
            response = qdrant.query_points(collection_name=coll, query=query_vec, limit=top_k)
        except Exception:
            continue
        for hit in response.points:
            scored_hits.append((hit.score, hit, coll))

    # 여러 컬렉션에서 모인 결과를 점수 기준으로 정렬해 상위 top_k개만 채택
    scored_hits.sort(key=lambda x: x[0], reverse=True)
    scored_hits = scored_hits[:top_k]

    results = []
    for score, hit, coll in scored_hits:
        entry = {
            "score": score,
            "text": hit.payload["text"],
            "source_document": collection_display_name(coll),  # 여러 문서를 섞어 검색할 때 출처 구분용
            "related_facts": [],
        }

        entity_ids = hit.payload.get("entity_ids")
        if neo4j_ok and entity_ids:
            try:
                entry["related_facts"] = run_cypher(
                    """
                    MATCH (a:Entity)-[r:RELATES]-(b:Entity)
                    WHERE a.canonical_id IN $ids
                    RETURN a.name AS source, r.type AS relation, b.name AS target
                    LIMIT 10
                    """,
                    {"ids": entity_ids},
                )
            except Exception:
                pass

        results.append(entry)
    return results


# ---- 에이전트 tool-use 로직 ----
AGENT_TOOLS = [
    {
        "name": "search_document",
        "description": "문서 내용을 벡터 유사도 검색(및 가능하면 그래프 관계 확장)으로 조회합니다. "
                        "사용자가 '전체 문서 통합 검색'을 선택했다면 모든 문서를 함께 검색합니다.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "검색할 질의문"}},
            "required": ["query"],
        },
    }
]


def run_tool(name, tool_input):
    if name == "search_document":
        try:
            return json.dumps(hybrid_search(tool_input["query"]), ensure_ascii=False)
        except Exception as ex:
            # 임베딩 API 장애 등으로 검색이 실패해도 앱이 죽지 않고 에이전트가 사정을 알리도록 함
            return json.dumps({"error": f"검색에 실패했습니다: {ex}"}, ensure_ascii=False)
    return "알 수 없는 도구입니다."


def ask_agent(question: str, history: list):
    messages = history + [{"role": "user", "content": question}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=3000,
            tools=AGENT_TOOLS,
            system=(
                "너는 업로드된 문서(한글 또는 영어) 기반 질의응답 에이전트야. "
                "search_document 도구로 문서 내용을 조회한 뒤, 조회된 내용에 근거해서만 답변해. "
                "문서가 영어라도 특별한 지시가 없으면 한국어로 답변해. "
                "검색 결과에 source_document 필드가 있으면, 여러 문서를 함께 검색한 상황이니 "
                "어떤 내용이 어느 문서에서 나왔는지 답변에서 구분해서 알려줘 (예: 'Tax Free Bagel 문서에 따르면...'). "
                "근거가 부족하면 모른다고 답해."
            ),
            messages=messages,
        )

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            return "\n".join(b.text for b in response.content if b.type == "text")

        messages.append({"role": "assistant", "content": response.content})
        tool_results = [
            {"type": "tool_result", "tool_use_id": b.id, "content": run_tool(b.name, b.input)}
            for b in tool_use_blocks
        ]
        messages.append({"role": "user", "content": tool_results})


# ---- 채팅 UI ----
st.title("📄 마케팅 지식그래프 기반 AI 에이전트")
st.caption("업로드된 문서의 지식그래프(Neo4j)와 벡터 검색(Qdrant)을 기반으로 답변합니다.")
st.info(f"👈 왼쪽 상단 **≫** 아이콘을 눌러 사이드바를 열면 **질문할 문서**를 고를 수 있어요. "
        f"(현재 선택: **{collection_display_name(COLLECTION_NAME)}**)")

if "messages" not in st.session_state:
    st.session_state.messages = []


def handle_question(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("답변을 찾는 중..."):
            answer = ask_agent(question, [])
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


# 버튼 클릭 시 즉시 처리하지 않고, 다음 실행에서 chat_input과 동일한 경로로 처리하기 위해
# session_state에 임시 저장한 뒤 재실행합니다 (메시지 중복 렌더링 방지).
pending_question = st.session_state.pop("pending_question", None)

if not st.session_state.messages:
    st.markdown("**💡 이렇게 질문해보세요**")

    if COLLECTION_NAME == ALL_DOCS_LABEL:
        example_questions = {
            "여러 문서의 공통 인사이트": "업로드된 문서들에서 공통적으로 발견할 수 있는 인사이트는 무엇인가요?",
            "예산 대비 성과 비교": "각 문서의 캠페인은 예산 대비 어떤 성과를 거뒀는지 비교해줘.",
            "실행 전략 비교": "각 캠페인의 실행 전략에는 어떤 차이가 있나요?",
        }
    else:
        example_questions = {
            "핵심 내용 요약": "이 문서의 핵심 내용을 요약해줘.",
            "문제와 해결 전략": "어떤 문제 상황이 있었고, 어떤 전략으로 해결했나요?",
            "주요 성과": "이 캠페인(또는 문서)의 주요 성과는 무엇인가요?",
            "관련 조직·브랜드 관계": "이 문서에 등장하는 주요 브랜드·조직·인물 간의 관계를 설명해줘.",
        }

    for label, q in example_questions.items():
        if st.button(f"👉 {label}", key=f"example_{label}", use_container_width=True):
            st.session_state["pending_question"] = q
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

typed_question = st.chat_input("문서에 대해 질문해보세요")
question = typed_question or pending_question

if question:
    handle_question(question)
