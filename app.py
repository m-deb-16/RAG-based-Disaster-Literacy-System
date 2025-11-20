"""
Main Streamlit Application for Disaster Literacy RAG System
Includes UI for all modes and admin panel for KB management
References: Lines 198-208 (Streamlit Frontend), Lines 209-231 (Admin Panel)
"""

import sys
from pathlib import Path

# Add src/backend to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "backend"))

import streamlit as st
from typing import Dict, Any, List
import time
import importlib # Import importlib

import config # Import config directly
from config import SAFETY_DISCLAIMER, SUPPORTED_MODES, DEFAULT_MODE
from controller import DisasterRAGController
from kb_manager import KBManager
from vector_store import VectorStore
from error_handler import error_handler

# Page configuration
st.set_page_config(
    page_title="Disaster Literacy RAG System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize Streamlit session state"""
    if 'controller' not in st.session_state:
        st.session_state.controller = None
    if 'online_mode' not in st.session_state:
        st.session_state.online_mode = False
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = "google"  # Default to Google
    if 'offline_model_mode' not in st.session_state:
        st.session_state.offline_model_mode = "economy"  # Default to economy
    if 'enable_translation' not in st.session_state:
        st.session_state.enable_translation = False  # Default to disabled
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = DEFAULT_MODE
    if 'simulation_data' not in st.session_state:
        st.session_state.simulation_data = None
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'kb_manager' not in st.session_state:
        st.session_state.kb_manager = KBManager()
    if 'show_stats' not in st.session_state:
        st.session_state.show_stats = False


def render_sidebar():
    """
    Render sidebar with mode selection
    References: Lines 200-201 (Online/Offline and Mode selection)
    """
    st.sidebar.title("🚨 Disaster Literacy RAG")
    st.sidebar.markdown("---")
    
    # Online/Offline Mode Selection
    st.sidebar.subheader("System Mode")
    online_mode = st.sidebar.radio(
        "Select Connection Mode:",
        ["Offline (Local LLM)", "Online (API LLM)"],
        index=0 if not st.session_state.online_mode else 1,
        help="Offline mode uses local models. Online mode requires API keys."
    )
    # Determine the selected mode
    is_online = (online_mode == "Online (API LLM)")
    
    # Only show LLM provider selection when in online mode
    if is_online:
        st.sidebar.subheader("Online LLM Provider")
        llm_provider = st.sidebar.radio(
            "Select LLM Provider:",
            ["Google Gemini", "Qwen via OpenRouter"],
            index=0 if st.session_state.llm_provider == "google" else 1,
            help="Choose which online LLM provider to use"
        )
        # Update provider in session state
        st.session_state.llm_provider = "google" if llm_provider == "Google Gemini" else "openrouter"
        
        # Show translation toggle only for Google Gemini
        if st.session_state.llm_provider == "google":
            st.session_state.enable_translation = st.sidebar.checkbox(
                "Enable Translation",
                value=st.session_state.enable_translation,
                help="Enable multilingual support (detects and translates non-English queries). Uses extra API calls."
            )
    else:
        # Show offline model mode selection when in offline mode
        st.sidebar.subheader("Offline Model Mode")
        offline_model = st.sidebar.radio(
            "Select Model Mode:",
            ["Economy (Lower spec systems)", "Power (Higher spec systems)"],
            index=0 if st.session_state.offline_model_mode == "economy" else 1,
            help="Economy mode uses Llama-2-7B (recommended for lower spec systems). Power mode uses Qwen2-7B (more capable, requires higher specs)."
        )
        # Update offline model mode in session state
        st.session_state.offline_model_mode = "economy" if offline_model == "Economy (Lower spec systems)" else "power"
        # Set provider to default when in offline mode
        st.session_state.llm_provider = "google"  # This won't matter in offline mode
    
    # Initialize/Reinitialize Controller
    if st.sidebar.button("🔄 Initialize System", use_container_width=True):
        with st.spinner("Initializing system..."):
            try:
                # Update session state immediately before initialization
                st.session_state.online_mode = is_online
                
                # Reload config to ensure latest .env values are used
                importlib.reload(config)
                # Set translation in config before creating controller
                config.ENABLE_TRANSLATION = st.session_state.enable_translation
                st.session_state.controller = DisasterRAGController(
                    online_mode=is_online,  # Pass the current selection directly
                    provider=st.session_state.llm_provider,
                    offline_model_mode=st.session_state.offline_model_mode
                )
                st.sidebar.success("✓ System initialized!")
            except Exception as e:
                st.sidebar.error(f"❌ Initialization failed: {e}")
    
    st.sidebar.markdown("---")
    
    # Operation Mode Selection
    st.sidebar.subheader("Operation Mode")
    
    # Filter modes based on online status
    available_modes = SUPPORTED_MODES.copy()
    if not is_online:
        if "Simulation" in available_modes:
            available_modes.remove("Simulation")
            
    st.session_state.current_mode = st.sidebar.radio(
        "Select Mode:",
        available_modes,
        help="Advisory: Get safety advice | Educational: Learn about disasters | Simulation: Test your knowledge (Online only)"
    )
    
    # Mode descriptions
    mode_descriptions = {
        "Advisory": "Get immediate safety guidance and action checklists",
        "Educational": "Learn about disaster preparedness and response",
        "Simulation": "Test your knowledge with interactive scenarios"
    }
    st.sidebar.info(mode_descriptions[st.session_state.current_mode])
    
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio("Navigate:", ["Main App", "Admin Panel"])
    
    return page


def render_advisory_mode():
    """Render Advisory mode interface"""
    st.header("🛡️ Advisory Mode")
    st.markdown("Get concise, actionable disaster safety advice")
    
    disaster_type = st.selectbox(
        "Disaster Type:",
        ["Any", "Tsunami", "Flood", "Cyclone", "Earthquake", "Fire", "Landslide"],
        help="Filter results by disaster type"
    )
    
    user_query = st.text_area(
        "What do you need guidance on?",
        placeholder="Example: What should I do during a tsunami warning?",
        height=100
    )
    
    if st.button("Get Advice", type="primary", use_container_width=True):
        if not st.session_state.controller:
            st.error("Please initialize the system first from the sidebar!")
            return
        
        if not user_query.strip():
            st.warning("Please enter your query")
            return
        
        with st.spinner("Retrieving safety guidance..."):
            # Convert to lowercase for case-insensitive matching
            disaster_filter = disaster_type.lower() if disaster_type != "Any" else None
            result = st.session_state.controller.process_query(
                user_query=user_query,
                mode="Advisory",
                disaster_type=disaster_filter
            )
        
        display_response(result)


def render_educational_mode():
    """Render Educational mode interface"""
    st.header("📚 Educational Mode")
    st.markdown("Learn about disaster preparedness and response")
    
    disaster_topic = st.text_input(
        "What would you like to learn about?",
        placeholder="Example: Tsunami safety measures",
        help="Enter a topic related to disaster preparedness"
    )
    
    if st.button("Learn", type="primary", use_container_width=True):
        if not st.session_state.controller:
            st.error("Please initialize the system first from the sidebar!")
            return
        
        if not disaster_topic.strip():
            st.warning("Please enter a topic")
            return
        
        with st.spinner("Preparing educational content..."):
            result = st.session_state.controller.process_query(
                user_query=disaster_topic,
                mode="Educational",
                disaster_type=None
            )
        
        display_response(result)


def render_simulation_mode():
    """
    Render Simulation mode interface
    References: Lines 202-207, 232-240 (Simulation workflow)
    """
    st.header("🎯 Simulation Mode")
    st.markdown("Test your disaster preparedness knowledge with interactive scenarios")
    
    if st.session_state.simulation_data is None:
        # Generate new simulation
        disaster_scenario = st.text_input(
            "Simulation Topic:",
            placeholder="Example: Coastal tsunami scenario",
            help="Describe the disaster scenario you want to practice"
        )
        
        if st.button("Start Simulation", type="primary", use_container_width=True):
            if not st.session_state.controller:
                st.error("Please initialize the system first from the sidebar!")
                return
            
            if not disaster_scenario.strip():
                st.warning("Please enter a scenario")
                return
            
            with st.spinner("Generating simulation scenario and questions..."):
                result = st.session_state.controller.process_query(
                    user_query=f"Create a simulation for: {disaster_scenario}",
                    mode="Simulation",
                    disaster_type=None
                )
            
            if result.get("success"):
                st.session_state.simulation_data = result["response"]
                st.session_state.user_answers = {}
                st.rerun()
    
    else:
        # Display simulation
        sim_data = st.session_state.simulation_data
        
        # Display scenario
        st.subheader("📖 Scenario")
        st.markdown(sim_data.get("scenario", ""))
        
        st.markdown("---")
        
        # Display questions
        st.subheader("❓ Questions")
        
        for mcq in sim_data.get("questions", []):
            q_num = mcq["question_num"]
            
            with st.container():
                st.markdown(f"**Question {q_num}:** {mcq['question']}")
                
                # Radio buttons for options
                answer = st.radio(
                    f"Select your answer for Question {q_num}:",
                    options=list(mcq["options"].keys()),
                    format_func=lambda x: f"{x}) {mcq['options'][x]}",
                    key=f"q_{q_num}",
                    index=None
                )
                
                if answer:
                    st.session_state.user_answers[q_num] = answer
                
                st.markdown("---")
        
        # Submit answers
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Submit Answers", type="primary", use_container_width=True):
                if len(st.session_state.user_answers) < len(sim_data.get("questions", [])):
                    st.warning("Please answer all questions before submitting")
                else:
                    # Check answers
                    score_result = st.session_state.controller.check_simulation_answers(
                        st.session_state.user_answers,
                        sim_data["questions"]
                    )
                    display_simulation_results(score_result)
        
        with col2:
            if st.button("New Simulation", use_container_width=True):
                st.session_state.simulation_data = None
                st.session_state.user_answers = {}
                st.rerun()


def display_response(result: Dict[str, Any]):
    """Display response from controller"""
    if result.get("error_occurred"):
        st.error(result.get("message", "An error occurred"))
        
        if "content" in result:
            st.markdown("### Available Guidance:")
            st.markdown(result["content"])
        
        if "suggestions" in result:
            st.markdown("### Suggested Actions:")
            for suggestion in result["suggestions"]:
                st.markdown(f"- {suggestion}")
    
    elif result.get("success"):
        response_data = result["response"]
        
        # Display main content
        st.markdown("### Response:")
        st.markdown(response_data.get("content", ""))
        
        # Display cited sources
        if "cited_sources" in response_data and response_data["cited_sources"]:
            with st.expander("📚 Sources Referenced"):
                for source in response_data["cited_sources"]:
                    st.text(f"• {source}")
        
        # Display action items (Advisory mode)
        if "action_items" in response_data and response_data["action_items"]:
            with st.expander("✅ Action Checklist"):
                for item in response_data["action_items"]:
                    st.markdown(f"- {item}")
        
        # Display metadata
        with st.expander("ℹ️ Response Details"):
            st.text(f"Mode: {result.get('mode')}")
            st.text(f"Response Time: {result.get('response_time', 0):.2f}s")
            st.text(f"Chunks Retrieved: {len(result.get('retrieved_chunks', []))}")


def display_simulation_results(score_result: Dict[str, Any]):
    """Display simulation test results"""
    st.markdown("---")
    st.subheader("📊 Your Results")
    
    # Score summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Questions", score_result["total_questions"])
    with col2:
        st.metric("Correct", score_result["correct_answers"], delta=None)
    with col3:
        st.metric("Incorrect", score_result["incorrect_answers"], delta=None)
    with col4:
        score_pct = score_result["score_percentage"]
        st.metric("Score", f"{score_pct:.1f}%")
    
    # Pass/Fail
    if score_result["passed"]:
        st.success(f"✅ Passed! You scored {score_pct:.1f}% (Pass threshold: {score_result['pass_threshold']}%)")
    else:
        st.error(f"❌ Not passed. You scored {score_pct:.1f}% (Pass threshold: {score_result['pass_threshold']}%)")
    
    # Detailed results
    st.markdown("### Detailed Results:")
    
    for res in score_result["results"]:
        with st.expander(f"Question {res['question_num']}: {'✅ Correct' if res['is_correct'] else '❌ Incorrect'}"):
            st.markdown(f"**Question:** {res['question']}")
            st.markdown(f"**Your Answer:** {res['user_answer']}")
            st.markdown(f"**Correct Answer:** {res['correct_answer']}")
            st.markdown(f"**Explanation:** {res['justification']}")


def render_admin_panel():
    """
    Render admin panel for KB management
    References: Lines 209-231 (Admin KB Updates)
    """
    st.header("⚙️ Admin Panel - Knowledge Base Management")
    
    tab1, tab2, tab3 = st.tabs(["Upload Documents", "KB Statistics", "System Info"])
    
    with tab1:
        st.subheader("Upload New Documents")
        st.markdown("Upload disaster preparedness documents to expand the knowledge base")

        # File uploader (multiple files)
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True
        )

        # Disaster category selection (applies to all files)
        disaster_type = st.selectbox(
            "Disaster Type (applies to all uploaded files):",
            ["Tsunami", "Flood", "Cyclone", "Earthquake", "Fire", "Landslide", "General"],
            help="Choose one category for all files, or use Auto-detect."
        )

        kb_manager = st.session_state.kb_manager
        
        # Container for processing results
        results_placeholder = st.empty()

        if uploaded_files and st.button("Process All Documents", type="primary", use_container_width=True):

            if not st.session_state.controller:
                st.error("Please initialize the system first from the sidebar.")
                return

            vector_store = st.session_state.controller.vector_store

            final_disaster_type = None if disaster_type == "Auto-detect" else disaster_type.lower()

            # Track processing results
            successful_uploads = []
            duplicate_uploads = []
            failed_uploads = []
            
            with results_placeholder.container():
                with st.spinner("Processing all documents... This may take time for scanned PDFs or large files."):
                    for uploaded_file in uploaded_files:

                        try:
                            # Save temporary file
                            save_path = Path("temp_upload") / uploaded_file.name
                            save_path.parent.mkdir(exist_ok=True)

                            with open(save_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                            # Process (OCR + chunk + metadata) with duplicate detection
                            result = kb_manager.process_document(
                                str(save_path),
                                disaster_type=final_disaster_type
                            )

                            # Check if duplicate
                            if result.get("is_duplicate"):
                                duplicate_uploads.append({
                                    "filename": uploaded_file.name,
                                    "reason": result.get("message")
                                })
                                st.warning(f"⚠️ Skipped (Duplicate): {uploaded_file.name}")
                                st.caption(result.get("message"))
                            elif result.get("success"):
                                # Add to vector store
                                chunks = result["chunks"]
                                vector_store.add_chunks(chunks)
                                successful_uploads.append(uploaded_file.name)
                                st.success(f"✅ Processed: {uploaded_file.name} ({len(chunks)} chunks)")
                            else:
                                failed_uploads.append({
                                    "filename": uploaded_file.name,
                                    "error": result.get("message", "Unknown error")
                                })
                                st.error(f"❌ Failed: {uploaded_file.name}")
                            
                            # Remove temp file
                            if save_path.exists():
                                save_path.unlink()
                        
                        except Exception as e:
                            failed_uploads.append({
                                "filename": uploaded_file.name,
                                "error": str(e)
                            })
                            st.error(f"❌ Error processing {uploaded_file.name}: {e}")
                            # Clean up temp file on error
                            if save_path.exists():
                                save_path.unlink()
                
                # Display summary
                st.markdown("---")
                st.subheader("📊 Upload Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("✅ Successful", len(successful_uploads))
                with col2:
                    st.metric("⚠️ Duplicates", len(duplicate_uploads))
                with col3:
                    st.metric("❌ Failed", len(failed_uploads))
                
                if successful_uploads:
                    st.info("♻️ Click 'Initialize System' in the sidebar to load the new documents into the RAG pipeline.")
                
                # Show details if there were duplicates or failures
                if duplicate_uploads:
                    with st.expander("⚠️ View Duplicate Details"):
                        for dup in duplicate_uploads:
                            st.text(f"• {dup['filename']}: {dup['reason']}")
                
                if failed_uploads:
                    with st.expander("❌ View Failure Details"):
                        for fail in failed_uploads:
                            st.text(f"• {fail['filename']}: {fail['error']}")


    with tab2:
        st.subheader("Knowledge Base Statistics")
        
        # Callback for deletion
        def delete_doc_callback(doc_id):
            print(f"DEBUG: Delete callback triggered for {doc_id}")
            if not st.session_state.controller:
                st.error("Initialize system first!")
                return
                
            try:
                # Delete from metadata
                source_name = st.session_state.kb_manager.delete_document(doc_id)
                
                if source_name:
                    # Delete from vector store
                    st.session_state.controller.vector_store.delete_document(source_name)
                    
                    # Force re-initialization
                    st.session_state.kb_manager = KBManager()
                    st.success(f"Deleted {source_name}")
                else:
                    st.error("Document not found")
            except Exception as e:
                st.error(f"Deletion failed: {e}")

        if st.button("Refresh Stats"):
            st.session_state.show_stats = True
            
        if st.session_state.show_stats:
            try:
                metadata = st.session_state.kb_manager.get_metadata()
                docs = st.session_state.kb_manager.list_documents()
                
                st.metric("Total Documents", len(docs))
                st.metric("Last Updated", metadata.get("last_updated", "Never"))
                
                if docs:
                    st.markdown("### Document List:")
                    for doc in docs:
                        with st.expander(f"📄 {doc['source']}"):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.text(f"Disaster Type: {doc['disaster_type']}")
                                st.text(f"Region: {doc['region']}")
                                st.text(f"Chunks: {doc['chunk_count']}")
                                st.text(f"Processed: {doc['processed_at']}")
                            
                            with col2:
                                st.button(
                                    "Delete", 
                                    key=f"del_{doc['id']}", 
                                    type="primary",
                                    on_click=delete_doc_callback,
                                    args=(doc['id'],)
                                )
                else:
                    st.info("No documents found.")
                
            except Exception as e:
                st.error(f"Failed to load stats: {e}")
    
    with tab3:
        st.subheader("System Information")
        
        if st.session_state.controller:
            stats = st.session_state.controller.get_system_stats()
            
            st.json(stats)
        else:
            st.warning("Controller not initialized")


def main():
    """Main application entry point"""
    init_session_state()
    
    # Sidebar
    page = render_sidebar()
    
    # Display safety disclaimer
    st.warning(SAFETY_DISCLAIMER)
    
    if page == "Main App":
        # Main application
        if st.session_state.current_mode == "Advisory":
            render_advisory_mode()
        elif st.session_state.current_mode == "Educational":
            render_educational_mode()
        elif st.session_state.current_mode == "Simulation":
            render_simulation_mode()
    
    else:
        # Admin panel
        render_admin_panel()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Disaster Literacy RAG System v1.0 | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
