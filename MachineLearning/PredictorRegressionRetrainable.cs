using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Threading;

namespace MachineLearning
{
   /// <summary>
   /// Modello a regressione con retrain
   /// </summary>
   public abstract class PredictorRegressionRetrainable : PredictorRegression
   {
      #region Fields
      /// <summary>
      /// Contatore di retrain
      /// </summary>
      [NonSerialized]
      private int _retrainCount;
      /// <summary>
      /// Seme per le operazioni random
      /// </summary>
      [NonSerialized]
      private int _seed;
      #endregion
      #region Properties
      /// <summary>
      /// Numero massimo di tentativi di retrain del modello
      /// </summary>
      public int MaxRetrain { get; set; } = 1;
      /// <summary>
      /// Livello di validazione. 0 semplice fit; 1 fit con shuffle delle righe; > 1 validazione incrociata
      /// </summary>
      public int ValidationLevel { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public PredictorRegressionRetrainable() { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Seme operazioni random</param>
      public PredictorRegressionRetrainable(int? seed) : base(seed) { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public PredictorRegressionRetrainable(MachineLearningContext ml) : base(ml) { }
      /// <summary>
      /// Funzione di restituzione della migliore fra due valutazioni modello
      /// </summary>
      /// <param name="modelEvaluation1">Prima valutazione</param>
      /// <param name="modelEvaluation2">Seconda valutazione</param>
      /// <returns>La migliore delle due valutazioni</returns>
      /// <remarks>Tenere conto che le valutazioni potrebbero essere null</remarks>
      protected override object GetBestModelEvaluation(object modelEvaluation1, object modelEvaluation2)
      {
         var best = base.GetBestModelEvaluation(modelEvaluation1, modelEvaluation2);
         if (best == modelEvaluation2)
            _retrainCount = 0;
         return best;
      }
      /// <summary>
      /// Restituisce il modello effettuando il training. Da implementare nelle classi derivate
      /// </summary>
      /// <param name="dataView">Datidi training</param>
      /// <param name="cancellation">Token di annullamento</param>
      /// <returns>Il modello appreso</returns>
      protected override ITransformer GetTrainedModel(IDataView dataView, CancellationToken cancellation)
      {
         // Verifica numero di tentativi massimi di retrain raggiunto
         if (++_retrainCount > MaxRetrain)
            return null;
         var pipe = GetPipe();
         cancellation.ThrowIfCancellationRequested();
         if (pipe == null)
            return null;
         // Training con selezione del tipo di validazione
         if (ValidationLevel < 1) {
            var result = pipe.Fit(dataView);
            cancellation.ThrowIfCancellationRequested();
            return result;
         }
         else if (ValidationLevel == 1) {
            var data = ML.NET.Data.ShuffleRows(dataView, _seed++);
            cancellation.ThrowIfCancellationRequested();
            var result = pipe.Fit(data);
            cancellation.ThrowIfCancellationRequested();
            return result;
         }
         else {
            var result = ML.NET.MulticlassClassification.CrossValidate(dataView, pipe, ValidationLevel, Name, null, _seed++);
            cancellation.ThrowIfCancellationRequested();
            return result.Best().Model;
         }
      }
      #endregion
   }
}
