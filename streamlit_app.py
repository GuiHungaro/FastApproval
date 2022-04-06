#Importando as bibliotecas
import streamlit as st
import pandas as pd
import pycaret.classification as pcc
from pycaret.classification import *
from PIL import Image


#Layout do site
image = Image.open('fastapproval.png')
st.image(image, caption='')
st.title("Sistema de aprovação rápida de crédito")

if st.button("Iniciar o aplicativo"):

  #Carregando o df 
  df = pd.read_csv('dados_de_credito.csv')


  #Carregando o modelo de classificação
  s = pcc.setup(data = df, target = 'INADIMPLENCIA', silent = True, fix_imbalance = True)
  best = pcc.compare_models(cross_validation = False)
  predictions = pcc.predict_model(best, data=df, raw_score=True)


  #Criando função para plotar os gráficos
  @st.cache(suppress_st_warning=True)
  def plot_charts(model, plots, package):
    for i in plots:
        try:
            st.markdown(f"#### {i}")
            package.plot_model(model, i ,display_format="streamlit")
        except:
            st.write(f"Plot {i}")


  #Criando a predição de crédito
  with st.expander("Predição de crédito"):

    #Coletando dados para serem preditos
    salario = st.number_input("Salário", value=0)
    idade = st.number_input("Idade", value=0)
    valor_emprestimo = st.number_input("Valor empréstimo", value=0)


    if st.button("Realizar avaliação"):

      novo_df = df.append({'SALARIO':salario, 'IDADE':idade, 'EMPRESTIMO':valor_emprestimo}, ignore_index=True)

      #Realizando a predição
      nova_predicao = pcc.predict_model(best, data=novo_df, raw_score=True)
      st.write(nova_predicao.iloc[-1])

      #Exibindo o resultado da predição
      label = nova_predicao.Label.iloc[-1]
      if label == 0:
        st.subheader("Crédito aprovado") 

      else:
        st.subheader("Crédito reprovado")       


  #Criando o método de avaliação do modelo
  with st.expander("Métricas de avaliação do modelo:"):

    st.write('O melhor modelo para representar os seus dados é:')
    st.write(best)

    st.write('Comparativa entre predição do modelo e a coluna inadimplência:')
    st.write(predictions)

    st.write('Avaliação gráfica do Modelo:')

    plots_class = st.multiselect(label = "Métricas de Classificação", 
                                 options = ['auc','pr','confusion_matrix','error','class_report','boundary', 'feature'],
                                 default = ['auc','pr','confusion_matrix','error','class_report','boundary', 'feature'])

    plot_charts(best, plots_class, pcc)
