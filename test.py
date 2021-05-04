from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt

file = 'datasets/datareal.csv'

headers = pd.read_csv(file).columns.tolist()
df = pd.read_csv(file)

LastRow = df.tail(1)


df = df.head(int(len(df)))


uniqueVars = []
for col in df:
    uniqueVars.append(list(df[col].unique()))


ifString = False

if df.dtypes[0] != 'int64':
    # STRING TO INTEGERS
    for i, item in enumerate(headers):
        for j, _var in enumerate(uniqueVars):
            for z, _othervar in enumerate(uniqueVars[j]):
                df.loc[df[item] == _othervar, item] = int(z+1)

data = df.iloc[-1].tolist()
data = data[0:-1]
df = df.head(int(len(df) - 1))

#data['output'] = 2 #VALUE RESPONSE

Y_vars = []

for i in range(len(df.index)):
    Y_vars.append(df.iloc[i][headers[-1]])


clf = tree.DecisionTreeClassifier(criterion='entropy')



df = df.drop(headers[-1], 1)

depVar = headers[-1]
print(depVar)

headers.remove(headers[-1]) # Borrando el ultimo elemento del header (juego)


X = df 
Y = Y_vars

model = clf.fit(X, Y)

text_representation = tree.export_text(clf)
#print(text_representation)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf, feature_names=headers ,filled=True)
fig.savefig("decistion_tree.png")



prediction = clf.predict([data])

print(prediction[0])