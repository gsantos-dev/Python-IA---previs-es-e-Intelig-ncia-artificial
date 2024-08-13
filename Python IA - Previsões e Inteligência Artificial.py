import pandas as pd

from sklearn.preprocessing import LabelEncoder


tabela = pd.read_csv("clientes.csv")


codificador = LabelEncoder()


#Profissão
tabela["profissao"] = codificador.fit_transform(tabela["profissao"])

#Mix crédito
tabela["mix_credito"] = codificador.fit_transform(tabela["mix_credito"])

#Comportamento_pagamento
tabela["comportamento_pagamento"] = codificador.fit_transform(tabela["comportamento_pagamento"])




# x é quem a IA pode utilizar para fazer as previsões
# y é a IA tem que prever (no caso aqui é o score de crédito)


x = tabela.drop(columns=["score_credito", "id_cliente"])
y = tabela["score_credito"]


from sklearn.model_selection import train_test_split


x_treino, x_teste, y_treino, y_teste = train_test_split(x, y)


#IMPORTAR A IA

#Arvore de decisão
from sklearn.ensemble import RandomForestClassifier

#KNN --> Vizinhos proximos
from sklearn.neighbors import KNeighborsClassifier

#CRIAR A IA
modelo_arvoredecisao = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()

#TREINAR A IA
modelo_arvoredecisao.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)

#COMPARANDO QUAL MODELO É MELHOR
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste)


from sklearn.metrics import accuracy_score

#Importar novos clientes para prever

tabela_novos_clientes = pd.read_csv("novos_clientes.csv")


tabela_novos_clientes["profissao"] = codificador.fit_transform(tabela_novos_clientes["profissao"])

#Mix crédito
tabela_novos_clientes["mix_credito"] = codificador.fit_transform(tabela_novos_clientes["mix_credito"])

#Comportamento_pagamento
tabela_novos_clientes["comportamento_pagamento"] = codificador.fit_transform(tabela_novos_clientes["comportamento_pagamento"])


previsoes = modelo_arvoredecisao.predict(tabela_novos_clientes)

print(tabela_novos_clientes)
print(previsoes)