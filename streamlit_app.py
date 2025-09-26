import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import html
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from typing import Annotated
from langchain_core.tools import Tool
from typing_extensions import TypedDict
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from prompt_guard import DEFAULT_PROMPT_SAFETY_CHECKER

# Set page config
st.set_page_config(
    page_title="Dispute Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load environment variables (expects GOOGLE_API_KEY in .env)
load_dotenv()

# Initialize the LangGraph agent
@st.cache_resource
def initialize_agent():
    """Initialize the LangGraph agent for chat functionality"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY missing. Please set it in the environment or .env file.")
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = init_chat_model("google_genai:gemini-2.5-flash")
    
    class State(TypedDict):
        messages: Annotated[list, add_messages]
    
    graph_builder = StateGraph(State)
    python_repl = PythonREPL()
    
    repl_tool = Tool(
        name="python_repl",
        description="""A Python shell for analyzing CSV data. Use this to execute python commands for data analysis.

            DATASET CONTEXT:
            The CSV file "final_disputes.csv" contains dispute analysis data with columns:
            - dispute_id: Unique identifier (e.g., D001, D002)
            - predicted_category: Dispute type (DUPLICATE_CHARGE, FRAUD, REFUND_PENDING, FAILED_TRANSACTION, OTHERS)
            - confidence: Prediction confidence score (0.0 to 1.0)
            - explanation: Why this category was predicted
            - suggested_action: Recommended action (Auto-refund, Manual review, Ask for more info, Escalate to bank)
            - justification: Reasoning behind the suggested action

            Always start by loading data: import pandas as pd; df = pd.read_csv("final_disputes.csv")
            Input should be valid python command. Use print(...) to see output values.""",
        func=python_repl.run,
    )
    
    tools = [repl_tool]
    llm_with_tools = llm.bind_tools(tools)
    
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    
    return graph_builder.compile()

# Load data
@st.cache_data
def load_dispute_data():
    """Load and prepare dispute data"""
    try:
        df = pd.read_csv("final_disputes.csv")

        # Ensure timestamp column is datetime for filtering/visuals
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        else:
            # Create synthetic timestamps over the last 30 days if missing
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            timestamps = pd.date_range(start=start_date, end=end_date, periods=len(df))
            df['timestamp'] = timestamps

        # Ensure status column exists and has sensible defaults
        if 'status' in df.columns:
            df['status'] = df['status'].fillna('Open')
        else:
            statuses = ['Open', 'In Review', 'Resolved', 'Escalated']
            df['status'] = pd.Series(statuses * (len(df) // len(statuses) + 1))[:len(df)]

        return df
    except FileNotFoundError:
        # Create sample data if file doesn't exist
        sample_data = {
            'dispute_id': ['D001', 'D002', 'D003', 'D004', 'D005'],
            'predicted_category': ['DUPLICATE_CHARGE', 'FAILED_TRANSACTION', 'FRAUD', 'REFUND_PENDING', 'OTHERS'],
            'confidence': [0.95, 0.9, 0.88, 0.92, 0.75],
            'explanation': [
                'Multiple successful transactions with same amount found.',
                'Transaction marked as failed but amount deducted.',
                'Suspicious pattern detected in transaction behavior.',
                'Customer requested refund for cancelled service.',
                'Unable to categorize dispute clearly.'
            ],
            'suggested_action': ['Auto-refund', 'Manual review', 'Escalate to bank', 'Ask for more info', 'Manual review'],
            'justification': [
                'Duplicate charge detected, refund can be issued automatically.',
                'Needs reconciliation with bank before resolving.',
                'Potential fraud requires immediate escalation.',
                'Need additional documentation from customer.',
                'Requires human review for proper categorization.'
            ],
            'timestamp': pd.date_range(start='2025-08-26', end='2025-09-25', periods=5),
            'status': ['Resolved', 'In Review', 'Escalated', 'Open', 'In Review']
        }
        return pd.DataFrame(sample_data)


def apply_global_styles():
    """Apply global dark theme and layout styles"""
    st.markdown(
        """
        <style>
        :root,
        html,
        body,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"] {
            background-color: #050505 !important;
            color: #f5f5f5 !important;
        }

        [data-testid="stSidebar"] {
            background-color: #080808 !important;
            border-right: 1px solid #151515;
        }

        [data-testid="stSidebar"] * {
            color: #f5f5f5 !important;
        }

        [data-testid="stMain"] {
            background-color: transparent !important;
        }

        h1, h2, h3, h4, h5, h6, p, label, span {
            color: #f5f5f5 !important;
        }

        .stMarkdown a {
            color: #6ca8ff !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            display: flex;
        }

        .stTabs [data-baseweb="tab"] {
            background: #0f0f0f !important;
            color: #f5f5f5 !important;
            border-radius: 999px !important;
            padding: 0.75rem 1.5rem !important;
            border: 1px solid #1a1a1a !important;
            flex: 1 !important;
            justify-content: center;
            align-items: center;
            text-align: center;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #1f6feb, #2654c5) !important;
            color: #ffffff !important;
            border: none !important;
        }

        .chat-shell {
            background: #0f0f0f;
            border: 1px solid #1f1f1f;
            border-radius: 16px;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            height: 65vh;
            max-height: 65vh;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
        }

        .chat-history {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            padding-right: 0.5rem;
        }

        .chat-history.empty {
            justify-content: center;
            align-items: center;
            color: #7a7a7a;
        }

        .chat-history::-webkit-scrollbar {
            width: 6px;
        }

        .chat-history::-webkit-scrollbar-thumb {
            background: #2d2d2d;
            border-radius: 3px;
        }

        .chat-message {
            display: flex;
        }

        .chat-message.user {
            justify-content: flex-end;
        }

        .chat-message.assistant {
            justify-content: flex-start;
        }

        .chat-bubble {
            max-width: 80%;
            padding: 0.75rem 1rem;
            border-radius: 12px;
            line-height: 1.5;
            background: #1a1a1a;
            color: #f5f5f5;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
        }

        .chat-message.user .chat-bubble {
            background: linear-gradient(135deg, #1f6feb, #2654c5);
            color: #ffffff;
        }

        .chat-placeholder {
            color: #7a7a7a;
            text-align: center;
            font-size: 0.95rem;
        }

        div[data-testid="stChatInput"] {
            padding-top: 1rem;
            background: #050505 !important;
        }

        div[data-testid="stChatInput"] textarea {
            background: #111111 !important;
            color: #f5f5f5 !important;
            border: 1px solid #1f1f1f !important;
        }

        div[data-testid="stChatInput"] button {
            background: linear-gradient(135deg, #1f6feb, #2654c5) !important;
            border: none !important;
        }

        .stMetric label {
            color: #9ea1a5 !important;
        }

        .stMetric [data-testid="stMetricValue"] {
            color: #f5f5f5 !important;
        }

        .stDataFrame {
            background-color: #0f0f0f !important;
        }

        .stDataFrame div[role="gridcell"] {
            color: #f5f5f5 !important;
        }

        .stSpinner > div {
            border-top-color: #1f6feb !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_chat_history() -> str:
    """Generate HTML for chat history"""
    if not st.session_state.chat_history:
        return (
            "<div class='chat-history empty'>"
            "<span class='chat-placeholder'>👋 Start a conversation by asking a question about the disputes!</span>"
            "</div>"
        )

    bubbles = ["<div class='chat-history'>"]
    for message in st.session_state.chat_history:
        role_class = "user" if message["role"] == "user" else "assistant"
        safe_content = html.escape(message["content"]).replace("\n", "<br>")
        bubbles.append(
            f"<div class='chat-message {role_class}'><div class='chat-bubble'>{safe_content}</div></div>"
        )
    bubbles.append("</div>")
    return "".join(bubbles)

def chat_interface():
    """Chat interface for the agent"""
    st.header("🤖 Dispute Analysis Chat")
    st.write("Ask questions about the dispute data and get AI-powered analysis!")
    
    # Initialize agent
    graph = initialize_agent()

    # Display chat history in a fixed-height container
    st.markdown(
        f"""
        <div class="chat-shell">
            {render_chat_history()}
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Chat input at the bottom - this will be fixed at the bottom
    user_input = st.chat_input("Ask a question about the disputes...")
    
    if user_input:
        is_safe, metadata = DEFAULT_PROMPT_SAFETY_CHECKER.check_prompt(user_input)
        if not is_safe:
            blocked_text = f"⚠️ Request blocked: {metadata['message']}"
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": blocked_text})
            st.warning(blocked_text)
            st.rerun()

        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Get response from agent
        with st.spinner("Analyzing..."):
            initial_messages = [
                {
                    "role": "system", 
                    "content": """You are a data analysis assistant. Your job is to write Python code to analyze CSV data.

                    INSTRUCTIONS:
                    - Use the python_repl tool to execute code
                    - Start by loading the data as instructed in the tool description
                    - Write code to answer the user's question
                    - Show relevant data snippets and analysis results
                    - Keep code simple and focused
                    - No visualizations, just data analysis"""
                },
                {"role": "user", "content": user_input}
            ]
            
            response_content = ""
            for event in graph.stream({"messages": initial_messages}):
                for value in event.values():
                    response_content = value["messages"][-1].content
            
            # Add assistant response to history
            st.session_state.chat_history.append({"role": "assistant", "content": response_content})
        
        # Rerun to update the display
        st.rerun()

def analysis_interface(filtered_df: pd.DataFrame):
    """Analysis interface with visualizations and summary metrics."""
    st.header("📊 Dispute Analysis Dashboard")

    if filtered_df.empty:
        st.info("No disputes match the selected filters.")
        return

    total_disputes = len(filtered_df)
    avg_confidence = filtered_df['confidence'].mean() if total_disputes else 0.0
    avg_conf_display = f"{avg_confidence:.2f}" if total_disputes else "N/A"

    status_series = filtered_df['status'] if 'status' in filtered_df.columns else pd.Series(dtype=str)
    active_cases = status_series.isin(['Open', 'In Review']).sum()
    resolved_cases = status_series.isin(['Resolved', 'Closed']).sum()
    escalated_cases = (status_series == 'Escalated').sum()

    action_series = filtered_df['suggested_action'] if 'suggested_action' in filtered_df.columns else pd.Series(dtype=str)
    auto_refund_count = (action_series == 'Auto-refund').sum()
    auto_refund_rate = (auto_refund_count / total_disputes * 100) if total_disputes else 0.0
    manual_review_count = (action_series == 'Manual review').sum()

    kpi_cols = st.columns(7)
    kpi_cols[0].metric("Total Disputes", total_disputes)
    kpi_cols[1].metric("Active Cases", active_cases)
    kpi_cols[2].metric("Resolved / Closed", resolved_cases)
    kpi_cols[3].metric("Escalated", escalated_cases)
    kpi_cols[4].metric("Average Confidence", avg_conf_display)
    kpi_cols[5].metric("Auto-refund Eligible", auto_refund_count)
    kpi_cols[6].metric("Manual Review Queue", manual_review_count)

    chart_row1 = st.columns(2)
    with chart_row1[0]:
        st.subheader("📈 Dispute Trends Over Time")

        if 'timestamp' in filtered_df.columns:
            timeline_df = filtered_df.dropna(subset=['timestamp'])
            if not timeline_df.empty:
                timeline_data = timeline_df.groupby([
                    timeline_df['timestamp'].dt.date,
                    'predicted_category'
                ]).size().reset_index(name='count')
                timeline_data.columns = ['date', 'category', 'count']

                fig_timeline = px.bar(
                    timeline_data,
                    x='date',
                    y='count',
                    color='category',
                    template="plotly_dark",
                    title="Disputes Over Time by Category",
                    labels={'count': 'Number of Disputes', 'date': 'Date'}
                )
                fig_timeline.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Number of Disputes",
                    paper_bgcolor="#050505",
                    plot_bgcolor="#050505",
                    font_color="#f5f5f5",
                    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.caption("No timeline data available for the selected filters.")
        else:
            st.caption("Timestamp information is not available for this dataset.")

    with chart_row1[1]:
        st.subheader("📌 Case Status Mix")
        if 'status' in filtered_df.columns:
            status_counts = filtered_df['status'].value_counts()
            if status_counts.empty:
                st.caption("No status data available for the selected filters.")
            else:
                status_counts_df = status_counts.reset_index()
                status_counts_df.columns = ['status', 'count']

                fig_status = px.bar(
                    status_counts_df,
                    x='status',
                    y='count',
                    text_auto=True,
                    template="plotly_dark",
                    title="Status Distribution"
                )
                fig_status.update_layout(
                    paper_bgcolor="#050505",
                    plot_bgcolor="#050505",
                    font_color="#f5f5f5",
                    xaxis_title="Status",
                    yaxis_title="Number of Disputes",
                    showlegend=False
                )
                st.plotly_chart(fig_status, use_container_width=True)
        else:
            st.caption("Status information is not available for this dataset.")

    chart_row2 = st.columns(2)
    with chart_row2[0]:
        st.subheader("📊 Dispute Categories")
        category_counts = filtered_df['predicted_category'].value_counts()
        if not category_counts.empty:
            fig_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                template="plotly_dark",
                title="Category Breakdown"
            )
            fig_pie.update_layout(
                paper_bgcolor="#050505",
                plot_bgcolor="#050505",
                font_color="#f5f5f5",
                legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.caption("No category data available for the selected filters.")

    with chart_row2[1]:
        st.subheader("🧭 Suggested Actions")
        if 'suggested_action' in filtered_df.columns:
            action_counts = filtered_df['suggested_action'].value_counts()
            if action_counts.empty:
                st.caption("No suggested action data available for the selected filters.")
            else:
                action_df = action_counts.reset_index()
                action_df.columns = ['suggested_action', 'count']

                fig_action = px.bar(
                    action_df,
                    x='suggested_action',
                    y='count',
                    text_auto=True,
                    template="plotly_dark",
                    title="Recommended Next Steps"
                )
                fig_action.update_layout(
                    paper_bgcolor="#050505",
                    plot_bgcolor="#050505",
                    font_color="#f5f5f5",
                    xaxis_title="Suggested Action",
                    yaxis_title="Number of Disputes",
                    showlegend=False
                )
                st.plotly_chart(fig_action, use_container_width=True)
        else:
            st.caption("Suggested action information is not available for this dataset.")

    chart_row3 = st.columns(1)
    with chart_row3[0]:
        st.subheader("🎯 Confidence by Category")
        confidence_by_category = (
            filtered_df.groupby('predicted_category')['confidence'].mean().sort_values()
            if 'predicted_category' in filtered_df.columns else pd.Series(dtype=float)
        )
        if not confidence_by_category.empty:
            conf_df = confidence_by_category.reset_index()
            conf_df.columns = ['predicted_category', 'avg_confidence']

            fig_conf = px.bar(
                conf_df,
                x='avg_confidence',
                y='predicted_category',
                orientation='h',
                template="plotly_dark",
                title="Average Model Confidence"
            )
            fig_conf.update_layout(
                paper_bgcolor="#050505",
                plot_bgcolor="#050505",
                font_color="#f5f5f5",
                xaxis_title="Confidence",
                yaxis_title="Category",
                showlegend=False
            )
            st.plotly_chart(fig_conf, use_container_width=True)
        else:
            st.caption("Confidence data by category is not available for the selected filters.")

def dispute_management_interface(df: pd.DataFrame, filtered_df: pd.DataFrame):
    """Interface for reviewing and updating individual disputes."""
    st.header("🛠️ Dispute Management")

    if filtered_df.empty:
        st.info("No disputes match the selected filters.")
        st.subheader("📋 Dispute Data Table")
        st.dataframe(filtered_df, use_container_width=True)
        return

    # Case History Tracking
    st.subheader("📝 Case History Tracking")

    dispute_options = filtered_df['dispute_id'].tolist()
    selected_dispute = st.selectbox(
        "Select Dispute ID for Details",
        dispute_options
    ) if dispute_options else None

    if selected_dispute:
        dispute_details = filtered_df[filtered_df['dispute_id'] == selected_dispute].iloc[0]

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Dispute Details:**")
            st.write(f"**ID**: {dispute_details['dispute_id']}")
            st.write(f"**Category**: {dispute_details['predicted_category']}")
            st.write(f"**Confidence**: {dispute_details['confidence']:.2f}")
            st.write(f"**Status**: {dispute_details.get('status', 'N/A')}")
            if 'timestamp' in dispute_details:
                timestamp_value = dispute_details['timestamp']
                if pd.notna(timestamp_value):
                    if isinstance(timestamp_value, pd.Timestamp):
                        formatted_date = timestamp_value.strftime('%Y-%m-%d')
                    else:
                        formatted_date = str(timestamp_value)
                    st.write(f"**Date**: {formatted_date}")

        with col2:
            st.write("**Analysis:**")
            st.write(f"**Explanation**: {dispute_details['explanation']}")
            st.write(f"**Suggested Action**: {dispute_details['suggested_action']}")
            st.write(f"**Justification**: {dispute_details['justification']}")

        st.markdown("---")

        existing_statuses = set(df['status'].dropna().unique().tolist()) if 'status' in df.columns else set()
        default_statuses = {"Open", "In Review", "Resolved", "Escalated"}
        status_options = sorted(existing_statuses.union(default_statuses)) or sorted(default_statuses)

        current_status = dispute_details.get('status', 'Open')
        if pd.isna(current_status) or current_status not in status_options:
            current_status = 'Open'

        with st.form(f"status_update_{selected_dispute}"):
            new_status = st.selectbox(
                "Update Dispute Status",
                options=status_options,
                index=status_options.index(current_status)
            )
            submitted = st.form_submit_button("Save Status")

        if submitted:
            if new_status == current_status:
                st.info("Status unchanged. No update required.")
            else:
                df.loc[df['dispute_id'] == selected_dispute, 'status'] = new_status
                filtered_df.loc[filtered_df['dispute_id'] == selected_dispute, 'status'] = new_status
                df.to_csv("final_disputes.csv", index=False)
                load_dispute_data.clear()
                st.success(f"Status for dispute {selected_dispute} updated to '{new_status}'.")
                st.rerun()

    # Data table
    st.subheader("📋 Dispute Data Table")
    st.dataframe(filtered_df, use_container_width=True)

def main():
    """Main Streamlit app"""
    apply_global_styles()
    st.title("🏦 Dispute Analysis Dashboard")

    df = load_dispute_data()

    # Sidebar filters shared across analysis and management views
    st.sidebar.header("Filters")

    categories = sorted(df['predicted_category'].dropna().unique().tolist()) if 'predicted_category' in df.columns else []
    default_categories = categories if categories else []
    selected_categories = st.sidebar.multiselect(
        "Select Categories",
        options=categories,
        default=default_categories,
    ) if categories else []

    if selected_categories:
        filtered_df = df[df['predicted_category'].isin(selected_categories)].copy()
    else:
        filtered_df = df.copy()

    date_range = None
    if 'timestamp' in df.columns:
        valid_timestamps = df['timestamp'].dropna()
        if not valid_timestamps.empty:
            min_date = valid_timestamps.min().date()
            max_date = valid_timestamps.max().date()
            default_range = [min_date, max_date]
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=default_range,
                min_value=min_date,
                max_value=max_date
            )

    if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2 and 'timestamp' in filtered_df.columns:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['timestamp'].dt.date >= start_date) & (filtered_df['timestamp'].dt.date <= end_date)]

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 CHAT", "📊 ANALYSIS", "🛠️ DISPUTE MANAGEMENT"])

    with tab1:
        chat_interface()

    with tab2:
        analysis_interface(filtered_df)

    with tab3:
        dispute_management_interface(df, filtered_df)

if __name__ == "__main__":
    main()