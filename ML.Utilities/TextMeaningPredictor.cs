using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace MachineLearning
{
   /// <summary>
   /// Modello per l'interpretazione del significato si testi
   /// </summary>
   [Serializable]
   public sealed class TextMeaningPredictor : Predictor
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public TextMeaningPredictor() => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Contesto di machine learning</param>
      public TextMeaningPredictor(int? seed) : base(seed) => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public TextMeaningPredictor(MachineLearningContext ml) : base(ml) => Init();
      /// Funzione di inizializzazione
      /// </summary>
      private void Init()
      {
         DataStorage = new DataStorageTextMemory();
         ModelStorage = new ModelStorageMemory();
      }
      /// <summary>
      /// Predizione
      /// </summary>
      /// <param name="data">Linea con i dati da usare per la previsione</param>
      /// <returns>Il task di predizione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public string Predict(string data) => PredictAsync(data, CancellationToken.None).Result;
      /// <summary>
      /// Predizione asincrona
      /// </summary>
      /// <param name="data">Elenco di dati da usare per la previsione</param>
      /// <returns>Il task di predizione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public string Predict(IEnumerable<string> data) => PredictAsync(data, CancellationToken.None).Result;
      /// <summary>
      /// Predizione asincrona
      /// </summary>
      /// <param name="data">Linea con i dati da usare per la previsione</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il task di predizione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public async Task<string> PredictAsync(string data, CancellationToken cancellation = default)
      {
         // Gestione avvio task di training
         _ = StartTrainingAsync(cancellation);
         // Crea una dataview con i dati di input
         var dataView = LoadData(new DataStorageTextMemory() { TextData = data, TextOptions = (DataStorage as IDataTextOptionsProvider)?.TextOptions ?? new TextLoader.Options() });
         cancellation.ThrowIfCancellationRequested();
         // Attande il modello od un eventuale errore di training
         var evaluation = await GetEvaluationAsync();
         cancellation.ThrowIfCancellationRequested();
         // Effettua la predizione
         var prediction = evaluation.Model.Transform(dataView);
         cancellation.ThrowIfCancellationRequested();
         // Restituisce il risultato
         return prediction.GetString("PredictedLabel");
      }
      /// <summary>
      /// Predizione asincrona
      /// </summary>
      /// <param name="data">Linea di dati da usare per la previsione</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il task di predizione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public Task<string> PredictAsync(IEnumerable<string> data, CancellationToken cancellation = default) => PredictAsync(FormatDataRow(data.ToArray()), cancellation);
      /// <summary>
      /// Predizione asincrona
      /// </summary>
      /// <param name="data">Linea di dati da usare per la previsione</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il task di predizione</returns>
      /// <remarks>La posizione corrispondente alla label puo' essere lasciata vuota</remarks>
      public Task<string> PredictAsync(CancellationToken cancellation = default, params string[] data) => PredictAsync(FormatDataRow(data), cancellation);
      #endregion
   }
}
