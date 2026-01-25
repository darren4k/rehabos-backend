"""Streamlit dashboard for RehabOS evaluation and monitoring."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="RehabOS Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_analytics():
    """Load analytics data."""
    from rehab_os.observability import PromptAnalytics

    analytics = PromptAnalytics(Path("data/logs"))
    end_time = datetime.now()
    start_time = end_time - timedelta(days=st.session_state.get("analysis_days", 7))
    return analytics.generate_report(start_time, end_time)


def load_feedback_stats():
    """Load feedback statistics."""
    feedback_file = Path("data/logs/feedback.jsonl")
    if not feedback_file.exists():
        return None

    entries = []
    with open(feedback_file) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return entries


def load_scheduler_tasks():
    """Load scheduled tasks."""
    tasks_file = Path("data/scheduler/tasks.json")
    if not tasks_file.exists():
        return []

    try:
        with open(tasks_file) as f:
            data = json.load(f)
            return data.get("tasks", [])
    except json.JSONDecodeError:
        return []


def render_sidebar():
    """Render the sidebar navigation."""
    st.sidebar.title("🏥 RehabOS")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        [
            "📊 Overview",
            "🤖 Agent Performance",
            "📝 Feedback Review",
            "🔧 Prompt Optimization",
            "⏰ Scheduled Tasks",
            "📈 Trends",
        ],
    )

    st.sidebar.markdown("---")

    # Time range selector
    st.sidebar.subheader("Analysis Period")
    days = st.sidebar.slider("Days", 1, 30, 7)
    st.session_state["analysis_days"] = days

    return page


def render_overview():
    """Render the overview dashboard."""
    st.title("📊 System Overview")

    try:
        report = load_analytics()
        report_dict = report.to_dict()
        summary = report_dict["summary"]

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Agent Runs",
                f"{summary['total_agent_runs']:,}",
            )

        with col2:
            st.metric(
                "LLM Calls",
                f"{summary['total_llm_calls']:,}",
            )

        with col3:
            st.metric(
                "Total Tokens",
                f"{summary['total_tokens']:,}",
            )

        with col4:
            st.metric(
                "Est. Cost",
                f"${summary['estimated_cost']:.2f}",
            )

        st.markdown("---")

        # Attention needed
        attention = report_dict.get("attention_needed", {})
        if any(attention.values()):
            st.subheader("⚠️ Attention Needed")

            col1, col2 = st.columns(2)

            with col1:
                if attention.get("high_correction_rate"):
                    st.warning(f"High correction rate: {', '.join(attention['high_correction_rate'])}")
                if attention.get("low_confidence"):
                    st.warning(f"Low confidence: {', '.join(attention['low_confidence'])}")

            with col2:
                if attention.get("high_latency"):
                    st.warning(f"High latency: {', '.join(attention['high_latency'])}")
                if attention.get("high_fallback"):
                    st.warning(f"High fallback: {', '.join(attention['high_fallback'])}")
        else:
            st.success("✅ All systems operating normally")

        st.markdown("---")

        # Agent summary table
        st.subheader("Agent Performance Summary")

        if report_dict["agents"]:
            agent_data = []
            for name, metrics in report_dict["agents"].items():
                agent_data.append({
                    "Agent": name,
                    "Runs": metrics["total_runs"],
                    "Success Rate": f"{metrics['success_rate']:.0%}",
                    "Avg Latency": f"{metrics['avg_latency_ms']:.0f}ms",
                    "Confidence": f"{metrics['avg_confidence']:.2f}" if metrics.get("avg_confidence") else "-",
                })

            st.dataframe(agent_data, use_container_width=True)
        else:
            st.info("No agent data available for the selected period.")

    except Exception as e:
        st.error(f"Failed to load analytics: {e}")
        st.info("Make sure the system has been running and generating logs.")


def render_agent_performance():
    """Render agent performance details."""
    st.title("🤖 Agent Performance")

    try:
        report = load_analytics()
        report_dict = report.to_dict()

        if not report_dict["agents"]:
            st.info("No agent data available.")
            return

        # Agent selector
        agent_names = list(report_dict["agents"].keys())
        selected_agent = st.selectbox("Select Agent", agent_names)

        if selected_agent:
            metrics = report_dict["agents"][selected_agent]

            st.markdown("---")

            # Metrics cards
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Runs", metrics["total_runs"])

            with col2:
                st.metric("Success Rate", f"{metrics['success_rate']:.0%}")

            with col3:
                st.metric("Avg Latency", f"{metrics['avg_latency_ms']:.0f}ms")

            with col4:
                conf = metrics.get("avg_confidence")
                st.metric("Confidence", f"{conf:.2f}" if conf else "N/A")

            st.markdown("---")

            # Additional metrics
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Quality Metrics")
                st.write(f"**Correction Rate:** {metrics.get('correction_rate', 0):.1%}")
                acceptance = metrics.get("acceptance_rate")
                st.write(f"**Acceptance Rate:** {acceptance:.1%}" if acceptance else "No feedback data")

            with col2:
                st.subheader("Token Usage")
                st.write(f"**Avg Tokens/Call:** {metrics.get('avg_tokens', 0):.0f}")

            # Model tier info
            tier_stats = report_dict.get("model_tiers", {})
            for tier, stats in tier_stats.items():
                if selected_agent in stats.get("agents", []):
                    st.info(f"This agent uses **{tier}** model tier")
                    break

    except Exception as e:
        st.error(f"Error loading agent data: {e}")


def render_feedback_review():
    """Render feedback review interface."""
    st.title("📝 Feedback Review")

    entries = load_feedback_stats()

    if not entries:
        st.info("No feedback data available yet.")
        st.markdown("""
        To generate feedback data:
        1. Use the API endpoint `POST /api/v1/feedback/annotate`
        2. Or review agent outputs through the feedback queue
        """)
        return

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)

    total = len(entries)
    accepted = sum(1 for e in entries if e.get("status") == "accepted")
    rejected = sum(1 for e in entries if e.get("status") == "rejected")
    needs_review = sum(1 for e in entries if e.get("status") == "needs_review")

    with col1:
        st.metric("Total Annotations", total)
    with col2:
        st.metric("Accepted", accepted, delta=f"{accepted/total:.0%}" if total else "0%")
    with col3:
        st.metric("Rejected", rejected)
    with col4:
        st.metric("Needs Review", needs_review)

    st.markdown("---")

    # By agent breakdown
    st.subheader("By Agent")

    by_agent = {}
    for entry in entries:
        agent = entry.get("agent_name", "unknown")
        if agent not in by_agent:
            by_agent[agent] = {"accepted": 0, "rejected": 0, "needs_review": 0}
        status = entry.get("status", "unknown")
        if status in by_agent[agent]:
            by_agent[agent][status] += 1

    agent_data = [
        {
            "Agent": agent,
            "Accepted": counts["accepted"],
            "Rejected": counts["rejected"],
            "Needs Review": counts["needs_review"],
            "Acceptance Rate": f"{counts['accepted'] / max(sum(counts.values()), 1):.0%}",
        }
        for agent, counts in by_agent.items()
    ]

    st.dataframe(agent_data, use_container_width=True)

    st.markdown("---")

    # Recent feedback
    st.subheader("Recent Feedback")

    recent = sorted(entries, key=lambda x: x.get("timestamp", ""), reverse=True)[:10]

    for entry in recent:
        with st.expander(f"{entry.get('agent_name')} - {entry.get('status')} ({entry.get('timestamp', '')[:10]})"):
            st.write(f"**Status:** {entry.get('status')}")
            if entry.get("notes"):
                st.write(f"**Notes:** {entry.get('notes')}")
            if entry.get("suggested_improvement"):
                st.write(f"**Suggested Improvement:** {entry.get('suggested_improvement')}")
            if entry.get("tags"):
                st.write(f"**Tags:** {', '.join(entry.get('tags', []))}")


def render_prompt_optimization():
    """Render prompt optimization interface."""
    st.title("🔧 Prompt Optimization")

    try:
        from rehab_os.learning.prompt_optimizer import PromptOptimizer

        optimizer = PromptOptimizer()

        # Optimization candidates
        st.subheader("Optimization Candidates")

        candidates = optimizer.get_optimization_candidates()

        if candidates:
            st.write(f"Found {len(candidates)} agents that could benefit from optimization:")

            for agent in candidates:
                with st.expander(f"🔴 {agent}"):
                    # Generate optimization suggestion
                    opt_feedback = optimizer.generate_optimization(agent, "feedback_based")
                    opt_metric = optimizer.generate_optimization(agent, "metric_based")

                    if opt_feedback:
                        st.write("**Feedback-based optimization:**")
                        st.write(f"- Reasoning: {opt_feedback.reasoning}")
                        st.write(f"- Changes: {', '.join(opt_feedback.changes)}")

                    if opt_metric:
                        st.write("**Metric-based optimization:**")
                        st.write(f"- Reasoning: {opt_metric.reasoning}")
                        st.write(f"- Changes: {', '.join(opt_metric.changes)}")

                    if st.button(f"Generate Optimized Prompt for {agent}", key=f"opt_{agent}"):
                        st.info("Prompt optimization would be generated here with LLM.")
        else:
            st.success("✅ No agents currently need optimization!")

        st.markdown("---")

        # Version history
        st.subheader("Prompt Version History")

        # List agents with registered prompts
        prompts_dir = Path("data/prompts")
        if prompts_dir.exists():
            prompt_files = list(prompts_dir.glob("*.txt"))
            if prompt_files:
                for pf in prompt_files[:10]:
                    st.write(f"- {pf.stem}")
            else:
                st.info("No prompt versions registered yet.")
        else:
            st.info("No prompt versions registered yet.")

    except ImportError as e:
        st.error(f"Missing dependencies: {e}")


def render_scheduled_tasks():
    """Render scheduled tasks management."""
    st.title("⏰ Scheduled Tasks")

    tasks = load_scheduler_tasks()

    if not tasks:
        st.info("No scheduled tasks configured.")

        if st.button("Setup Default Schedule"):
            try:
                from rehab_os.learning.scheduler import create_learning_scheduler

                scheduler = create_learning_scheduler()
                st.success("Default schedule configured!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to setup schedule: {e}")
        return

    # Tasks table
    st.subheader("Configured Tasks")

    task_data = []
    for task in tasks:
        task_data.append({
            "Task ID": task.get("task_id"),
            "Type": task.get("task_type"),
            "Schedule": task.get("schedule"),
            "Last Run": task.get("last_run", "Never")[:16] if task.get("last_run") else "Never",
            "Next Run": task.get("next_run", "N/A")[:16] if task.get("next_run") else "N/A",
            "Status": task.get("last_status", "pending"),
            "Enabled": "✅" if task.get("enabled") else "❌",
        })

    st.dataframe(task_data, use_container_width=True)

    st.markdown("---")

    # Task details
    st.subheader("Task Details")

    task_ids = [t.get("task_id") for t in tasks]
    selected_task = st.selectbox("Select Task", task_ids)

    if selected_task:
        task = next((t for t in tasks if t.get("task_id") == selected_task), None)
        if task:
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Type:** {task.get('task_type')}")
                st.write(f"**Schedule:** {task.get('schedule')}")
                st.write(f"**Run Count:** {task.get('run_count', 0)}")

            with col2:
                st.write(f"**Enabled:** {task.get('enabled')}")
                if task.get("last_error"):
                    st.error(f"Last Error: {task.get('last_error')}")

            st.write("**Configuration:**")
            st.json(task.get("config", {}))

            if task.get("last_result"):
                st.write("**Last Result:**")
                st.json(task.get("last_result"))


def render_trends():
    """Render trends and historical analysis."""
    st.title("📈 Trends")

    st.info("Trend analysis requires historical data over multiple days/weeks.")

    try:
        from rehab_os.observability import PromptAnalytics

        analytics = PromptAnalytics(Path("data/logs"))

        # Compare different time periods
        st.subheader("Period Comparison")

        col1, col2 = st.columns(2)

        with col1:
            period1_days = st.number_input("Period 1 (last N days)", 1, 30, 7)

        with col2:
            period2_days = st.number_input("Period 2 (previous N days)", 1, 30, 7)

        if st.button("Compare Periods"):
            end_time = datetime.now()

            # Period 1: recent
            start1 = end_time - timedelta(days=period1_days)
            report1 = analytics.generate_report(start1, end_time)

            # Period 2: previous
            start2 = start1 - timedelta(days=period2_days)
            report2 = analytics.generate_report(start2, start1)

            dict1 = report1.to_dict()
            dict2 = report2.to_dict()

            st.markdown("---")

            # Compare summaries
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"Recent ({period1_days} days)")
                st.write(f"Agent Runs: {dict1['summary']['total_agent_runs']}")
                st.write(f"LLM Calls: {dict1['summary']['total_llm_calls']}")
                st.write(f"Tokens: {dict1['summary']['total_tokens']:,}")

            with col2:
                st.subheader(f"Previous ({period2_days} days)")
                st.write(f"Agent Runs: {dict2['summary']['total_agent_runs']}")
                st.write(f"LLM Calls: {dict2['summary']['total_llm_calls']}")
                st.write(f"Tokens: {dict2['summary']['total_tokens']:,}")

            # Calculate changes
            if dict2['summary']['total_agent_runs'] > 0:
                runs_change = (dict1['summary']['total_agent_runs'] - dict2['summary']['total_agent_runs']) / dict2['summary']['total_agent_runs']
                st.metric("Agent Runs Change", f"{runs_change:+.0%}")

    except Exception as e:
        st.error(f"Error loading trend data: {e}")


def main():
    """Main dashboard entry point."""
    page = render_sidebar()

    if page == "📊 Overview":
        render_overview()
    elif page == "🤖 Agent Performance":
        render_agent_performance()
    elif page == "📝 Feedback Review":
        render_feedback_review()
    elif page == "🔧 Prompt Optimization":
        render_prompt_optimization()
    elif page == "⏰ Scheduled Tasks":
        render_scheduled_tasks()
    elif page == "📈 Trends":
        render_trends()


if __name__ == "__main__":
    main()
