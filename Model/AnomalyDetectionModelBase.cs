using MachineLearning.Data;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Text;
using TMetrics = Microsoft.ML.Data.AnomalyDetectionMetrics;
using TTrainers = MachineLearning.Trainers.AnomalyDetectionTrainers;

namespace MachineLearning.Model
{
   /// <summary>
   /// Classe base per i rilevatori di anomalie
   /// </summary>
   [Serializable]
   public abstract class AnomalyDetectionModelBase : ModelBaseMLNet
   {
      #region Properties
      /// <summary>
      /// Nome colonna label
      /// </summary>
      public string LabelColumnName { get; set; } = "Label";
      /// <summary>
      /// Catalogo di trainers
      /// </summary>
      [field: NonSerialized]
      public TTrainers Trainers { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider contesto di machine learning</param>
      public AnomalyDetectionModelBase(IContextProvider<MLContext> contextProvider = default) : base(contextProvider) =>
         Trainers = new TTrainers(this);
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
         if (modelEvaluation1 is TMetrics metrics1 && modelEvaluation2 is TMetrics metrics2)
            best = metrics2.DetectionRateAtFalsePositiveCount > metrics1.DetectionRateAtFalsePositiveCount ? modelEvaluation2 : modelEvaluation1;
         return best;
      }
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="model">Modello da valutare</param>
      /// <param name="data">Dati attuali caricati</param>
      /// <returns>Il risultato della valutazione</returns>
      /// <remarks>La valutazione ottenuta verra' infine passata alla GetBestEvaluation per compaare e selezionare il modello migliore</remarks>
      protected override object GetModelEvaluation(IDataTransformer model, IDataAccess data) =>
         Context.AnomalyDetection.Evaluate(model.Transform(data), LabelColumnName ?? "Label");
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="modelEvaluation">Il risultato della valutazione di un modello</param>
      /// <returns>Il risultato della valutazione in formato testo</returns>
      protected override string GetModelEvaluationInfo(object modelEvaluation)
      {
         if (modelEvaluation is not TMetrics metrics)
            return null;
         var sb = new StringBuilder();
         sb.AppendLine(metrics.ToText());
         return sb.ToString();
      }
      #endregion
   }
}
