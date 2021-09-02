using MachineLearning.Data;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using System;
using System.Linq;
using System.Text;
using System.Threading;
using TCalibratedMetrics = Microsoft.ML.Data.CalibratedBinaryClassificationMetrics;
using TExperimentSettings = Microsoft.ML.AutoML.BinaryExperimentSettings;
using TMetric = Microsoft.ML.AutoML.BinaryClassificationMetric;
using TMetrics = Microsoft.ML.Data.BinaryClassificationMetrics;
using TTrainers = MachineLearning.Trainers.BinaryClassificationTrainers;

namespace MachineLearning.Model
{
   /// <summary>
   /// Classe base per i classificatori binari
   /// </summary>
   [Serializable]
   public abstract class BinaryModelBase : ModelBaseMLNet, IModelTrainingAuto, IModelTrainingCrossValidate
   {
      #region Fields
      /// <summary>
      /// Evento di modello di autotraining disponibile
      /// </summary>
      [NonSerialized]
      private AutoTrainingTask<TMetrics, TExperimentSettings> autoTrainingTask;
      #endregion
      #region Properties
      /// <summary>
      /// Metrica di scelta del miglior modello
      /// </summary>
      public TMetric BestModelSelectionMetric { get; set; }
      /// <summary>
      /// Nome colonna label
      /// </summary>
      public string LabelColumnName { get; set; } = "Label";
      /// <summary>
      /// Catalogo di trainers
      /// </summary>
      public TTrainers Trainers { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider contesto di machine learning</param>
      public BinaryModelBase(IContextProvider<MLContext> contextProvider = default) : base(contextProvider) =>
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
         if (modelEvaluation1 is TCalibratedMetrics calibratedMetrics1 && modelEvaluation2 is TCalibratedMetrics calibratedMetrics2)
            best = calibratedMetrics2.Accuracy >= calibratedMetrics1.Accuracy && calibratedMetrics2.LogLoss < calibratedMetrics1.LogLoss ? modelEvaluation2 : modelEvaluation1;
         else if (modelEvaluation1 is TMetrics metrics1 && modelEvaluation2 is TMetrics metrics2)
            best = metrics2.Accuracy >= metrics1.Accuracy ? modelEvaluation2 : modelEvaluation1;
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
         Context.BinaryClassification.Evaluate(model.Transform(data), LabelColumnName ?? "Label");
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="modelEvaluation">Il risultato della valutazione di un modello</param>
      /// <returns>Il risultato della valutazione in formato testo</returns>
      protected override string GetModelEvaluationInfo(object modelEvaluation)
      {
         if (modelEvaluation is TCalibratedMetrics calibratedMetrics) {
            var sb = new StringBuilder();
            sb.AppendLine(calibratedMetrics.ToText());
            return sb.ToString();
         }
         else if (modelEvaluation is TMetrics metrics) {
            var sb = new StringBuilder();
            sb.AppendLine(metrics.ToText());
            return sb.ToString();
         }
         return null;
      }
      /// <summary>
      /// Effettua il training con la ricerca automatica del miglior trainer
      /// </summary>
      /// <param name="data">Dati</param>
      /// <param name="maxTimeInSeconds">Numero massimo di secondi di training</param>
      /// <param name="metrics">La metrica del modello migliore</param>
      /// <param name="numberOfFolds">Numero di validazioni incrociate</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il modello migliore</returns>
      IDataTransformer IModelTrainingAuto.AutoTraining(IDataAccess data, int maxTimeInSeconds, out object metrics, int numberOfFolds, CancellationToken cancellation)
      {
         autoTrainingTask ??= new AutoTrainingTask<TMetrics, TExperimentSettings>(this);
         var result = autoTrainingTask.WaitResult(
            () => Context.Auto().CreateBinaryClassificationExperiment(new TExperimentSettings
            {
               CancellationToken = cancellation,
               OptimizingMetric = BestModelSelectionMetric,
               MaxExperimentTimeInSeconds = (uint)Math.Max(0, maxTimeInSeconds),
            }),
            models => models.Best(),
            data,
            LabelColumnName,
            out var m,
            numberOfFolds,
            cancellation);
         metrics = m;
         return result;
      }
      /// <summary>
      /// Effettua il training con validazione incrociata del modello
      /// </summary>
      /// <param name="data">Dati</param>
      /// <param name="metrics">La metrica del modello migliore</param>
      /// <param name="numberOfFolds">Numero di validazioni</param>
      /// <param name="samplingKeyColumnName">Nome colonna di chiave di campionamento</param>
      /// <param name="seed">Seme per le operazioni random</param>
      /// <returns>Il modello migliore</returns>
      IDataTransformer IModelTrainingCrossValidate.CrossValidateTraining(IDataAccess data, out object metrics, int numberOfFolds, string samplingKeyColumnName, int? seed, CancellationToken cancellation)
      {
         var results = Context.BinaryClassification.CrossValidate(data, GetPipes().Merged, numberOfFolds, LabelColumnName ?? "Label", samplingKeyColumnName, seed);
         if (results == null) {
            metrics = null;
            return null;
         }
         cancellation.ThrowIfCancellationRequested();
         var best = (from r in results select (r.Model, r.Metrics)).Best();
         metrics = best.Metrics;
         cancellation.ThrowIfCancellationRequested();
         return new DataTransformer<MLContext>(this, best.Model);
      }
      #endregion
   }
}
