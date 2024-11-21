import streamlit as st
import ollama
import json
import time
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from graph_utils import plot_graph

def get_available_models():
    try:
        models = ollama.list()
        return [model['name'] for model in models['models']]
    except Exception as e:
        st.error(f"Failed to fetch models: {str(e)}")
        return ["llama3.2"]  # Return default model if fetching fails

def make_api_call(messages, max_tokens, model_name, is_final_answer=False):
    for attempt in range(3):
        try:
            response = ollama.chat(
                model=model_name,
                messages=messages,
                options={
                    "num_predict": max_tokens,
                    "temperature": 0.2
                }
            )
            
            print(f"Raw API response: {response}")
            
            if 'message' not in response or 'content' not in response['message']:
                raise ValueError(f"Unexpected API response structure: {response}")
            
            content = response['message']['content']
            done_reason = response.get('done', False)
            
            # Remove any content before the first step or final answer
            content = re.sub(r'^.*?((?:### )?Step 1:|### Final Answer:)', r'\1', content, flags=re.DOTALL)
            
            # Parse the multi-step response
            steps = re.split(r'((?:### )?Step \d+:.*?(?=\n)|### Final Answer:.*?(?=\n))', content, flags=re.DOTALL)
            steps = [step.strip() for step in steps if step.strip()]

            parsed_steps = []
            for i in range(0, len(steps), 2):
                if i + 1 < len(steps):
                    title = steps[i].strip()
                    content = steps[i+1].strip()
                    
                    if "Final Answer" in title:
                        next_action = "final_answer"
                    else:
                        if not title.startswith("###"):
                            title = f"### {title}"
                        next_action = "continue"
                    
                    parsed_steps.append({
                        "title": title,
                        "content": content,
                        "next_action": next_action
                    })

            # If we found valid steps, return them along with done_reason
            if parsed_steps:
                return parsed_steps, done_reason
            
            # If no valid steps found, create a single step from the entire content
            return [{
                "title": "### Response",
                "content": content,
                "next_action": "final_answer"
            }], done_reason

        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return [{"title": "### Error", "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"}], None
                else:
                    return [{"title": "### Error", "content": f"Failed to generate step after 3 attempts. Error: {str(e)}", "next_action": "final_answer"}], None
            time.sleep(1)  # Wait for 1 second before retrying

    return None, None

def get_embedding(text, model_name):
    response = ollama.embeddings(model=model_name, prompt=text)
    return np.array(response['embedding'])

def calculate_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

def find_strongest_path(G, start, end):
    def dfs(node, path, total_weight):
        if node == end:
            return path, total_weight
        
        best_path, best_weight = None, -float('inf')
        for neighbor in G.neighbors(node):
            if neighbor not in path:
                edge_weight = G[node][neighbor]['weight']
                new_path, new_weight = dfs(neighbor, path + [neighbor], total_weight + edge_weight)
                if new_path and new_weight > best_weight:
                    best_path, best_weight = new_path, new_weight
        
        return best_path, best_weight

    strongest_path, _ = dfs(start, [start], 0)
    if strongest_path:
        strongest_edges = list(zip(strongest_path[:-1], strongest_path[1:]))
        return strongest_path, strongest_edges
    return None, None


def generate_response(prompt, model_name, max_tokens):
    messages = [
        {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step, incorporating dynamic Chain of Thought (CoT), reflection, and verbal reinforcement learning. Follow these guidelines:

1. Structure your response with clear steps, each starting with "### Step X: [Step Title]" where X is the step number.
2. Use at least 5 steps in your reasoning BEFORE providing the final answer.
3. For each step, provide detailed content explaining your thought process.
4. Explore multiple angles and approaches in your reasoning.
5. After each step, decide if you need another step or if you're ready to give the final answer.
6. Continuously adjust your reasoning based on intermediate results and reflections, adapting your strategy as you progress.
7. Regularly evaluate your progress, being critical and honest about your reasoning process.
8. Assign a quality score between 0.0 and 1.0 to guide your approach:
   - 0.8+: Continue current approach
   - 0.5-0.7: Consider minor adjustments
   - Below 0.5: Seriously consider backtracking and trying a different approach
   - If writing out the quality score do so in the format "**Quality Score:** 0.8" for example.
9. If unsure or if your score is low, backtrack and try a different approach, explaining your decision.
10. For mathematical problems, show all work explicitly using LaTeX for formal notation and provide detailed proofs. Use $ for inline LaTeX and $$ for display LaTeX.
11. Explore multiple solutions individually if possible, comparing approaches in your reflections.
12. Write out all calculations and reasoning explicitly.
13. Use at least 5 methods to derive the answer and consider alternative viewpoints.
14. Be aware of your limitations as an AI and use best practices in your reasoning.
15. After every 3 steps, perform a detailed self-reflection on your reasoning so far, considering potential biases and alternative viewpoints.
16. End with a final step titled "### Final Answer:"
17. In the "### Final Answer:" step, provide a concise summary of your conclusion.

Example structure:
### Step 1: [Step Title]
[Detailed thought process, exploring multiple angles]
[Assign quality score]
[Step 1 content]

### Step 2: [Step Title]
[Detailed thought process, exploring multiple angles]
[Assign quality score]
[Step 2 content]

### Step 3: [Step Title]
[Detailed thought process, exploring multiple angles]
[Assign quality score]
[Step 3 content]

### Step 4: Self-Reflection
[Detailed self-reflection on reasoning so far]
[Consider potential biases and alternative viewpoints]
[Decide whether to continue or change approach]
[Self-reflection content]

[Continue with more steps...]

### Final Answer:
[Concise summary of the conclusion]

Remember to be thorough in your analysis and adapt your approach based on your ongoing reflections."""},
        {"role": "user", "content": prompt},
    ]
    
    reasoning_steps = []
    total_thinking_time = 0
    
    # Create graph
    G = nx.Graph()
    embeddings = []
    
    start_time = time.time()
    step_data_list, done_reason = make_api_call(messages, max_tokens, model_name)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    for i, step_data in enumerate(step_data_list):
        step_content = f"{step_data['title']}\n{step_data['content']}"
        reasoning_steps.append((step_data['title'].strip(), step_data['content'].strip(), thinking_time / len(step_data_list)))
        
        # Generate embedding for this step
        embedding = get_embedding(step_content, model_name)
        embeddings.append(embedding)
        
        # Add node to graph
        node_id = f"Step{i+1}"
        G.add_node(node_id, label=step_data['title'].replace("### ", ""))
        
        # Calculate similarities with previous steps
        for j in range(i):
            similarity = calculate_similarity(embedding, embeddings[j])
            if similarity > 0.5:  # Only add edges for high similarities
                G.add_edge(f"Step{j+1}", node_id, weight=similarity)
        
        if step_data['next_action'] == 'final_answer':
            # Calculate strongest path
            if len(G.nodes()) > 1:
                try:
                    strongest_path, strongest_edges = find_strongest_path(G, "Step1", node_id)
                    print(f"Strongest edges: {strongest_edges}")  # Add this line for debugging
                except Exception as e:
                    print(f"Error finding strongest path: {e}")
                    strongest_path, strongest_edges = None, None
            else:
                strongest_path, strongest_edges = None, None

            yield reasoning_steps, (step_data['title'], step_data['content'], thinking_time), total_thinking_time, done_reason, G, strongest_edges
            return
    
    # This line should not be reached, but just in case:
    yield reasoning_steps, None, total_thinking_time, done_reason, G, None

def render_latex(content):
    # Replace "### Quality Score:" with "**Quality Score:**"
    content = content.replace("### Quality Score:", "**Quality Score:**")
    
    # Escape colons that are not part of LaTeX commands
    content = re.sub(r'(?<!\\\w):(?!\s*\\\w)', '\\:', content)
    
    # Split the content into LaTeX and non-LaTeX parts
    parts = re.split(r'(\\\[.*?\\\]|\$\$.*?\$\$|\$.*?\$)', content, flags=re.DOTALL)
    rendered_parts = []
    for part in parts:
        if part.startswith('\\[') and part.endswith('\\]'):
            # Render display LaTeX
            rendered_parts.append(st.latex(part.strip('\\[]')))
        elif part.startswith('$$') and part.endswith('$$'):
            # Render display LaTeX
            rendered_parts.append(st.latex(part.strip('$')))
        elif part.startswith('$') and part.endswith('$'):
            # Render inline LaTeX
            rendered_parts.append(st.latex(part.strip('$')))
        elif part.strip():
            # Render regular text
            rendered_parts.append(st.markdown(part))
    return rendered_parts


def main():
    st.set_page_config(page_title="o1lama", page_icon="🦙", layout="wide")
    
    st.title("o1lama")
    
    st.markdown("Using Ollama to create reasoning chains that run locally and are similar in appearance to o1.")
    
    # Get available models and create a dropdown menu
    available_models = get_available_models()
    selected_model = st.selectbox("Select a model:", available_models)
    
    # Add dropdown for token selection with 1024 as default
    token_options = [512, 1024, 2048, 4096]
    selected_tokens = st.selectbox("Select max tokens:", token_options, index=token_options.index(1024))

    # Add dropdown for layout selection
    # layout_options = ['force', 'circular', 'spectral', 'kamada_kawai']
    # selected_layout = st.selectbox("Select graph layout:", layout_options, index=0)
    selected_layout = 'circular'  # Hard-code it to circular layout
    
    # Text area for user query (4 lines high)
    user_query = st.text_area("Enter your query:", placeholder="e.g., How many times does the letter 'R' appear in the word 'strawberry'?", height=120)
    
    # Create placeholder containers
    response_container = st.empty()
    time_container = st.empty()
    graph_container = st.empty()
    
    if user_query:
        # Clear previous response
        response_container.empty()
        time_container.empty()
        graph_container.empty() 
        
        # Show "Generating response..." message with a spinner
        with st.spinner("Generating response..."):
            final_reasoning_steps = []
            final_answer = None
            final_done_reason = None
            final_graph = None
            final_strongest_edges = None  # Changed from final_strongest_path
            for reasoning_steps, answer, total_thinking_time, done_reason, graph, strongest_edges in generate_response(user_query, selected_model, selected_tokens):
                final_reasoning_steps = reasoning_steps
                final_done_reason = done_reason
                final_graph = graph
                final_strongest_edges = strongest_edges  # Changed from final_strongest_path
                if answer:
                    final_answer = answer

        with response_container.container():
            if len(final_reasoning_steps) > 1:  # Check if there are multiple steps
                st.markdown("### Reasoning")
                for step in final_reasoning_steps[:-1]:  # Exclude the last step
                    with st.expander(step[0], expanded=True):
                        render_latex(step[1])
            
            if final_answer:
                st.markdown("### Final Answer")  # Display "Final Answer" without colon
                render_latex(final_answer[1])
            elif final_reasoning_steps:  # If there's no final answer but there are steps
                render_latex(final_reasoning_steps[-1][1])
            else:  # If there are no steps and no final answer
                st.markdown("No detailed reasoning steps were provided.")


        # Display the graph in its own container
        with graph_container.container():
            if final_graph:
                st.subheader("Knowledge Graph")
                fig = plot_graph(final_graph, final_strongest_edges, layout_type=selected_layout)
                st.plotly_chart(fig)

        # Show total time
        if total_thinking_time is not None:
            time_container.markdown(f"**Total thinking time: {total_thinking_time:.2f} seconds**")
        
        # Display warning if response was truncated due to token limit
        if final_done_reason == "length":
            st.warning("The response was truncated due to token limit. Consider increasing the max token value for a more complete response.")


if __name__ == "__main__":
    main()

