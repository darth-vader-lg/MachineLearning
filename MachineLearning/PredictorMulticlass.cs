using MachineLearning.Trainers;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Text;

namespace MachineLearning
{
   /// <summary>
   /// Classe base per i previsori di tipo multiclasse
   /// </summary>
   public abstract class PredictorMulticlass : Predictor
   {
      #region Properties
      /// <summary>
      /// Catalogo di trainers
      /// </summary>
      [field: NonSerialized]
      public MulticlassClassificationTrainersCatalog Trainers { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public PredictorMulticlass() : base() => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Seme operazioni random</param>
      public PredictorMulticlass(int? seed) : base(seed) => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public PredictorMulticlass(MachineLearningContext ml) : base(ml) => Init();
      /// <summary>
      /// Funzione di restituzione della migliore fra due valutazioni modello
      /// </summary>
      /// <param name="modelEvaluation1">Prima valutazione</param>
      /// <param name="modelEvaluation2">Seconda valutazione</param>
      /// <returns>La migliore delle due valutazioni</returns>
      /// <remarks>Tenere conto che le valutazioni potrebbero essere null</remarks>
      protected override object GetBestModelEvaluation(object modelEvaluation1, object modelEvaluation2)
      {
         var best = modelEvaluation2;
         if (modelEvaluation1 is MulticlassClassificationMetrics metrics1 && modelEvaluation2 is MulticlassClassificationMetrics metrics2)
            best = metrics2.MicroAccuracy >= metrics1.MicroAccuracy && metrics2.LogLoss < metrics1.LogLoss ? modelEvaluation2 : modelEvaluation1;
         return best;
      }
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="model">Modello da valutare</param>
      /// <param name="data">Dati attuali caricati</param>
      /// <returns>Il risultato della valutazione</returns>
      /// <remarks>La valutazione ottenuta verra' infine passata alla GetBestEvaluation per compaare e selezionare il modello migliore</remarks>
      protected override object GetModelEvaluation(ITransformer model, IDataView data) => ML.NET.MulticlassClassification.Evaluate(model.Transform(data));
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="modelEvaluation">Il risultato della valutazione di un modello</param>
      /// <returns>Il risultato della valutazione in formato testo</returns>
      protected override string GetModelEvaluationInfo(object modelEvaluation)
      {
         if (modelEvaluation is not MulticlassClassificationMetrics metrics)
            return null;
         var sb = new StringBuilder();
         sb.AppendLine(metrics.ToText());
         sb.AppendLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
         return sb.ToString();
      }
      /// <summary>
      /// Funzione di inizializzazione
      /// </summary>
      private void Init() => Trainers = new MulticlassClassificationTrainersCatalog(ML);
      #endregion
   }
}
