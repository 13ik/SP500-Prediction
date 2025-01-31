# Predykcja Ruchów Indeksu S&P 500 za pomocą Uczenia Maszynowego  

---

## Cel Projektu  
Celem projektu jest stworzenie modelu uczenia maszynowego do przewidywania ruchów indeksu giełdowego S&P 500. Dzięki analizie danych historycznych oraz wykorzystaniu zaawansowanych algorytmów, system może wspierać podejmowanie decyzji inwestycyjnych.  

---

## Dane  
W projekcie wykorzystano zbiór danych zawierający historyczne informacje o cenach akcji spółek wchodzących w skład indeksu S&P 500 oraz dodatkowe wskaźniki rynkowe.  

**Główne cechy danych:**  
- **Date** – data notowań  
- **Cena zamknięcia wybranych spółek** (np. ADBE, INTC, MSFT, NVDA, TSLA, itp.)  
- **Zmienna docelowa (`sp500_increase`)**:
  - **-1** – spadek indeksu  
  - **0** – brak wyraźnej zmiany  
  - **1** – wzrost indeksu  

---

## Analiza Danych (EDA)  
Przeprowadzono eksploracyjną analizę danych, w tym:  
- Rozkład klas zmiennej docelowej  
- Macierz korelacji między cechami  
- Wizualizację rozkładu wybranych zmiennych  

---

## Przygotowanie Danych  
1. **Selekcja cech** – wybrano istotne zmienne wpływające na ruch indeksu  
2. **Standaryzacja danych** – zastosowano `StandardScaler` do przeskalowania cech  
3. **Podział na zbiory treningowy i testowy** – 80% danych wykorzystano do treningu, 20% do testowania  

---

## Trenowanie Modeli  
Przetestowano trzy algorytmy:  
- **Random Forest Classifier**  
- **Gradient Boosting Classifier**  
- **Logistic Regression**  

Dla modeli przeprowadzono optymalizację hiperparametrów za pomocą `GridSearchCV`.  

---

## Ocena Modeli  
Modele oceniono za pomocą metryk:  
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1 Score**  
- **Macierz pomyłek**  

Najlepszy wynik osiągnął **Random Forest Classifier**, uzyskując najwyższą skuteczność w przewidywaniu ruchów indeksu.  

---

## Zapis Modelu  
Najlepszy model oraz skaler zapisano w formacie `.pkl` przy użyciu `joblib`, umożliwiając ich późniejsze wykorzystanie.  

---

## Testowanie Modelu  
Zaimplementowano testy jednostkowe, aby sprawdzić poprawność działania modelu:  
- Test wczytywania modelu  
- Test poprawności predykcji  

---

## Wnioski  
- **Random Forest Classifier** jest najbardziej efektywnym modelem dla tego problemu  
- **Standaryzacja danych** poprawia wyniki modelu logistycznego  
- **Feature importance** pozwala lepiej zrozumieć wpływ poszczególnych zmiennych na ruch indeksu  

---

## Wdrożenie  
Model może zostać wykorzystany w aplikacji do przewidywania przyszłych ruchów indeksu S&P 500 na podstawie nowych danych rynkowych.  
