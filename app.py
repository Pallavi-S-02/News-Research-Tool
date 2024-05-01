import os
import streamlit as st
import pickle
import time
from news_research_tool import news_data, load_llm, load_url_data, check_url_access, create_chunks_and_embeddings, create_prompt_template
from news_research_tool import file_path, model_name

def main():
    st.title('News Research Tool 📈')
    st.sidebar.title('News Article URLs')
    urls = news_data()
    process_url_clicked = st.sidebar.button("Process URLs")
    main_placeholder = st.empty()
    stop_processing = False
    if process_url_clicked:
        data = load_url_data(urls,main_placeholder)
        check_url_access(data,main_placeholder)
        if not stop_processing:
            create_chunks_and_embeddings(data,main_placeholder)
    if not stop_processing:
        query = main_placeholder.text_input("Question: ")
        if query:
            if os.path.exists(file_path):
                with open(file_path,'rb') as f:
                    vectorstore = pickle.load(f)
                    llm = load_llm(model_name)
                    retrieval_chain = create_prompt_template(llm, vectorstore)
                    start = time.process_time()
                    response = retrieval_chain.invoke({"input": query})
                    print(f"Response time: {time.process_time() - start}")
                    st.header("Answer")
                    st.write(response["answer"])
                    # print('response', response)
                    # printed_links = set()  
                    # st.subheader("Sources : ")
                    # for doc in response['context']:
                    #     if 'source' in doc.metadata:
                    #         source_link = doc.metadata['source']
                    #         if source_link not in printed_links:
                    #             printed_links.add(source_link)
                    #             print(f"Source Link: {source_link}")
                    #             st.write(source_link)



main()

