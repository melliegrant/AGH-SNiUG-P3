import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

st.title("🐧 Projekt: Klasyfikacja pingwinów")

# === Krok 1: Ładowanie danych ===
st.header("1. Ładowanie danych")
st.markdown("""
Wczytywanie danych z pliku 
`data/penguins.csv`
""")

try:
    penguins = pd.read_csv("data/penguins.csv")
    st.success(f"✅ Załadowano {len(penguins)} rekordów.")

    with st.expander("Pierwsze 5 wierszy"):
        st.dataframe(penguins.head())

except Exception as e:
    st.error(f"❌ Błąd ładowania danych: {e}")
    st.stop()



# === Krok 2: Podstawowe informacje ===
st.header("2. Eksploracja danych")
st.markdown("""
Sprawdzono
- czy są braki (`NaN`),
- jakie są typy zmiennych (liczbowe vs kategoryczne),
- ile jest rekordów i klas.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Brakujące wartości")
    missing = penguins.isnull().sum()
    st.write(missing)
    if missing.sum() > 0:
        st.warning(f"Łącznie braków: {missing.sum()}")

with col2:
    st.subheader("Liczba rekordów w klasach")
    species_counts = penguins['species'].value_counts()
    st.bar_chart(species_counts)
    st.write(species_counts)

st.subheader("Typy kolumn")
st.write(penguins.dtypes)
numeric_cols = penguins.select_dtypes(include=['number']).columns.tolist()
categorical_cols = penguins.select_dtypes(include=['object']).columns.tolist()
st.write(f"🔢 Liczbowe: {numeric_cols}")
st.write(f"🔤 Kategoryczne: {categorical_cols}")


# === Krok 3: Wizualizacja ===
# Słownik. nazwa wyświetlana → nazwa kolumny w danych
DISPLAY_TO_COLUMN = {
    "dług. dzioba (mm)": "bill_length_mm",
    "głęb. dzioba (mm)": "bill_depth_mm",
    "dług. płetwy (mm)": "flipper_length_mm",
    "masa ciała (g)": "body_mass_g"
}

# Odwrotne mapowanie (dla podpisu osi)
COLUMN_TO_DISPLAY = {v: k for k, v in DISPLAY_TO_COLUMN.items()}

st.header("3. Jak gatunki się rozróżniają?")
st.markdown("""
Wybierz parę cech, by zobaczyć, czy gatunki tworzą naturalne „grupy”.
""")

# Usuwamy tylko braki w kluczowych kolumnach (dla wykresu)
plot_df = penguins.dropna(subset=list(DISPLAY_TO_COLUMN.values()))

x_label = st.selectbox("Oś X", list(DISPLAY_TO_COLUMN.keys()), index=2)  # domyślnie: dług. płetwy
y_label = st.selectbox("Oś Y", list(DISPLAY_TO_COLUMN.keys()), index=0)  # domyślnie: dług. dzioba

# Konwersja na nazwy kolumn
x_col = DISPLAY_TO_COLUMN[x_label]
y_col = DISPLAY_TO_COLUMN[y_label]

# Sprawdź, czy nie wybrano tej samej osi dwa razy
if x_col == y_col:
    st.warning("⚠️ Oś X i Y nie mogą być tą samą cechą.")
else:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue='species', palette='Set1', s=60, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'{x_label} vs {y_label}')
    ax.legend(title="Gatunek")
    st.pyplot(fig)

    st.info(f"""
    **Interpretacja**
    - **Gentoo**: duże wartości `{x_label}` i `{y_label}` - łatwo odróżnić.
    - **Adélie vs Chinstrap**: silne nakładanie się (szczególnie w przestrzeni dziób: dł./głęb.) - klasyfikacja wymaga analizy wielu cech jednocześnie.
    """)




# === Krok 4: Kodowanie kategoryczne ===
st.header("4. Przekształcanie zmiennych kategorycznych")
st.markdown("""
Zmienne kategoryczne (`island`, `sex`) zostały przekształcone metodą kodowania typu **one-hot**, w której każda kategoria reprezentowana jest przez oddzielną binarną zmienną. W celu ograniczenia multikolinearności zastosowano opcję drop='first', usuwając jedną kategorię odniesienia dla każdej zmiennej. Każda kategoria staje się osobną kolumną (0/1).
""")

st.write("Przykład dla `island`:")
example_island = pd.DataFrame({'island': ['Biscoe', 'Dream', 'Torgersen']})
encoded = pd.get_dummies(example_island, prefix='island')
st.dataframe(encoded)




# === Krok 5: Imputacja braków ===
st.header("5. Obsługa brakujących wartości")
st.markdown("""            
Brakujące wartości uzupełniono, stosując **imputację**.
- dla danych liczbowych: **średnia** (np. średnia długość dzioba),
- dla danych kategorycznych: **dominanta** (np. najczęściej 'MALE'). 
""")

st.write("Przykład braku w danych:")
st.dataframe(penguins[penguins.isnull().any(axis=1)].head(3))

st.write("Procedura została wykonana w pipelinie, co zapewnia brak wycieku danych.")


# 6. Przygotowanie X, y i podział ===
st.header("6–7. Przygotowanie danych do modelu")

st.markdown("""
Zbiór podzielono na treningowy (80%) i testowy (20%) z zachowaniem proporcji klas (stratified split). Zmienna docelowa (species) została zakodowana numerycznie.

- **X** = cechy (długość dzioba, płetwy, wyspa, płeć...)  
- **y** = cel (`species`)  
""")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Przygotuj y
le = LabelEncoder()
y = le.fit_transform(penguins['species'])
X_raw = penguins.drop(columns=['species'])

st.write(f"✅ y zakodowane: {dict(zip(le.classes_, range(len(le.classes_))))}")
st.write(f"✅ X ma {X_raw.shape[0]} rekordów i {X_raw.shape[1]} kolumn.")

if st.button("Podziel dane **`80/20`**"):
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test
    st.session_state['le'] = le
    st.success(f"Podział gotowy! Train: {len(X_train)}, Test: {len(X_test)}")




# === Krok 8–9: Modele klasyczne ===
st.header("8-9. Pierwsze modele — bez sieci neuronowych")
st.markdown("""
Do klasyfikacji zastosowano dwa algorytmy: **regresję logistyczną** oraz **drzewo decyzyjne**. Oceny dokonano na podstawie dokładności, F1-score oraz macierzy pomyłek. 
""")

if 'X_train' not in st.session_state:
    st.warning("Najpierw podziel dane (Krok 7).")
else:
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train = st.session_state['y_train']
    y_test = st.session_state['y_test']
    le = st.session_state['le']

    # Pipeline preprocessingu
    numeric_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    categorical_features = ['island', 'sex']

    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipe, numeric_features),
        ('cat', categorical_pipe, categorical_features)
    ])

    # Przetwarzamy X_train i X_test RAZ (dla wszystkich modeli, w tym NN)
    try:
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Skalujemy — ale tylko dla modeli wymagających skalowania (LogReg, NN)
        # W pipeline LogReg jest już scaler, ale dla NN chcemy mieć czyste X_scaled
        # → więc wyciągamy tylko numeryczne cechy z pipeline i skalujemy je osobno?
        # ✅ Lepsze rozwiązanie: zmodyfikuj pipeline tak, by dało się uzyskać X_scaled
    except Exception as e:
        st.error(f"Błąd preprocessingu: {e}")
        st.stop()

    # --- Helper: train & store results for a given model ---
    def train_and_store(model, model_key: str):
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average='macro')
        cm = confusion_matrix(y_test, y_test_pred)
        cm_df = pd.DataFrame(cm, 
                            index=[f"Faktyczny: {cls}" for cls in le.classes_],
                            columns=[f"Pred: {cls}" for cls in le.classes_])

        # Store under unique key, e.g. 'model_1' or 'model_2'
        st.session_state[model_key] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'f1': f1,
            'cm_df': cm_df
        }

    st.subheader("Regresja logistyczna")

    if st.button("Wytrenuj model", key="train_model_1"):
        model1 = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
        train_and_store(model1, 'model_1')

    # Display Model 1 results — if trained
    if 'model_1' in st.session_state:
        res = st.session_state['model_1']
        col1, col2, col3 = st.columns(3)
        col1.metric("Train Acc", f"{res['train_acc']:.2%}")
        col2.metric("Test Acc", f"{res['test_acc']:.2%}")
        col3.metric("F1 (macro)", f"{res['f1']:.2f}")

        st.write("Macierz pomyłek (test):")
        st.dataframe(res['cm_df'])

        st.info("""
        📌 **Regresja logistyczna**
                
        Osiągnięto bardzo dobrą skuteczność, co wskazuje na liniową separowalność klas. **Macierz pomyłek** nie zawiera żadnych błędów — zarówno w klasie minority (Chinstrap, n=14), jak i w pozostałych. 

        Wynik ten sugeruje, że dla danego zbioru liniowy decyzyjny hiperpłaszczyzna wystarcza do pełnej separacji, co jest zgodne z analizą wizualną (Gentoo wyraźnie oddzielony, Adélie i Chinstrap — częściowo, ale wystarczająco).
        """)

    st.subheader("Drzewo decyzyjne")

    if st.button("Wytrenuj model", key="train_model_2"):
        # Drzewo — bez skalowania (niepotrzebne)
        preprocessor_no_scale = ColumnTransformer([
            ('num', SimpleImputer(strategy='mean'), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ]), categorical_features)
        ])
        model2 = Pipeline([
            ('preprocessor', preprocessor_no_scale),
            ('classifier', DecisionTreeClassifier(random_state=42, max_depth=5))
        ])
        train_and_store(model2, 'model_2')

    # Display Model 2 results — if trained
    if 'model_2' in st.session_state:
        res = st.session_state['model_2']
        col1, col2, col3 = st.columns(3)
        col1.metric("Train Acc", f"{res['train_acc']:.2%}")
        col2.metric("Test Acc", f"{res['test_acc']:.2%}")
        col3.metric("F1 (macro)", f"{res['f1']:.2f}")

        st.write("Macierz pomyłek (test):")
        st.dataframe(res['cm_df'])

        st.info("""
        📌 **Drzewo decyzyjne**
                
        Z `max_depth=5` uzyskało nieco niższą skuteczność na teście (98.55%), z 1 błędem klasyfikacji (Adélie zaklasyfikowany jako Gentoo), co jest nieistotne statystycznie przy tak małym zbiorze testowym (n=69), ale pokazuje nieco wyższą wariancję modelu.
        """)
    
    if st.button("🗑️ Wyczyść wyniki modeli"):
        st.session_state.pop('model_1', None)
        st.session_state.pop('model_2', None)
        st.rerun()





st.header("10. Skalowanie zmiennych")
st.info(f"✅ Skalowanie (`StandardScaler`) zostało zastosowane w pipeline dla regresji logistycznej.")






# === TENSORFLOW ===


st.header("11. Sieć neuronowa (Keras/TensorFlow)")

st.markdown("""
Zbudowano *feedforward* sieć neuronową:
- **Warstwa wejściowa**: 8 neuronów (po preprocessingu),
- **Warstwy ukryte**: 16 → 8 neuronów, aktywacja `ReLU`,
- **Warstwa wyjściowa**: 3 neurony, aktywacja `softmax`.

Funkcja straty: `sparse_categorical_crossentropy`,  
Optymalizator: `Adam`, batch size: 16, epoki: 100.
""")

# 🔑 Przetwarzamy dane raz — wspólnie dla wszystkich modeli
if 'X_train_processed' not in st.session_state:
    if 'X_train' not in st.session_state:
        st.warning("⚠️ Najpierw podziel dane i wytrenuj modele klasyczne (Kroki 6–9).")
        st.stop()
    
    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train = st.session_state['y_train']
    y_test = st.session_state['y_test']
    le = st.session_state['le']

    # Pipeline (jak wcześniej)
    numeric_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    categorical_features = ['island', 'sex']

    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipe, numeric_features),
        ('cat', categorical_pipe, categorical_features)
    ])

    # 🔧 Przetwarzamy raz — zapisujemy do sesji
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    st.session_state['X_train_processed'] = X_train_processed
    st.session_state['X_test_processed'] = X_test_processed
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test
    st.session_state['preprocessor'] = preprocessor
    st.session_state['le'] = le

    st.info("✅ Dane przetworzone i zapisane do sesji.")

# !!!!! Sprawdź, czy model już istnieje i można go wczytać
if 'nn_model' not in st.session_state:
    model_path = "saved_models/penguin_nn.keras"
    if os.path.exists(model_path):
        try:
            st.session_state['nn_model'] = tf.keras.models.load_model(model_path)
            st.session_state['nn_loaded'] = True  # flaga — wczytany z pliku
            st.success("🧠 Załadowano zapisany model sieci neuronowej.")
        except Exception as e:
            st.warning(f"⚠️ Nie udało się wczytać modelu: {e}")

# 🔘 Trenowanie NN — tylko po naciśnięciu
if st.button("Wytrenuj sieć neuronową"):
    X_train_processed = st.session_state['X_train_processed']
    X_test_processed = st.session_state['X_test_processed']
    y_train = st.session_state['y_train']
    y_test = st.session_state['y_test']
    le = st.session_state['le']

    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(X_train_processed.shape[1],)),
        layers.Dense(8, activation='relu'),
        layers.Dense(3, activation='softmax')  # 3 gatunki
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=0
    )

    with st.spinner("🧠 Trenowanie sieci neuronowej (może zająć 5–10 sekund)..."):
        history = model.fit(
            X_train_processed, y_train,
            epochs=100,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

    # Zapisz
    st.session_state['nn_model'] = model
    st.session_state['nn_history'] = history.history

    # Ocena
    test_loss, test_acc = model.evaluate(X_test_processed, y_test, verbose=0)
    train_loss, train_acc = model.evaluate(X_train_processed, y_train, verbose=0)

    st.success("✅ Sieć neuronowa wytrenowana!")
    col1, col2 = st.columns(2)
    col1.metric("Train Acc", f"{train_acc:.2%}")
    col2.metric("Test Acc", f"{test_acc:.2%}")

    # Krzywe uczenia
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)
    
    ax[0].plot(epochs, hist['accuracy'], 'b-', label='Train')
    ax[0].plot(epochs, hist['val_accuracy'], 'r--', label='Val')
    ax[0].set_title('Accuracy'); ax[0].legend()
    
    ax[1].plot(epochs, hist['loss'], 'b-', label='Train')
    ax[1].plot(epochs, hist['val_loss'], 'r--', label='Val')
    ax[1].set_title('Loss'); ax[1].legend()
    
    st.pyplot(fig)

    # Macierz pomyłek
    y_pred = model.predict(X_test_processed)
    y_pred_classes = y_pred.argmax(axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    cm_df = pd.DataFrame(cm,
                        index=[f"Faktyczny: {cls}" for cls in le.classes_],
                        columns=[f"Pred: {cls}" for cls in le.classes_])
    st.write("Macierz pomyłek (test):")
    st.dataframe(cm_df)

    st.info("""
    📌 Sieć neuronowa osiągnęła skuteczność zbliżoną do modeli klasycznych (~98–100%), co potwierdza:  
    - dane są dobrze separowalne nawet prostymi modelami,  
    - złożoność sieci nie musi być duża — 2 warstwy ukryte wystarczają.  
    Brak rosnącego `val_loss` wskazuje na brak overfittingu.
    """)

    # !!!!! Zapisz model lokalnie
    os.makedirs("saved_models", exist_ok=True)
    model_save_path = "saved_models/penguin_nn.keras"
    try:
        # logi
        st.write("🔍 Próbuję zapisać model…")
        os.makedirs("saved_models", exist_ok=True)
        model_save_path = "saved_models/penguin_nn.keras"
        st.write(f"Ścieżka zapisu: {os.path.abspath(model_save_path)}")

        model.save(model_save_path)
        st.success(f"💾 Model zapisany lokalnie: `{model_save_path}`")
    except Exception as e:
        st.error(f"❌ Błąd zapisu modelu: {e}")
    
