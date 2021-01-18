using MachineLearning.Data;
using MachineLearning.Trainers;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using System;
using System.Linq;
using System.Text;
using System.Threading;

namespace MachineLearning.Model
{
   /// <summary>
   /// Classe base per i previsori a regressione
   /// </summary>
   [Serializable]
   public abstract class RegressionModelBase : ModelBase
   {
      #region Properties
      /// <summary>
      /// Metrica di scelta del miglior modello
      /// </summary>
      public RegressionMetric BestModelSelectionMetric { get; set; } = RegressionMetric.RSquared; //@@@ Rendere serializzabile
      /// <summary>
      /// Catalogo di trainers
      /// </summary>
      [field: NonSerialized]
      public RegressionTrainersCatalog Trainers { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public RegressionModelBase() : base() => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Seme operazioni random</param>
      public RegressionModelBase(int? seed) : base(seed) => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public RegressionModelBase(MachineLearningContext ml) : base(ml) => Init();
      /// <summary>
      /// Effettua il training con la ricerca automatica del miglior trainer
      /// </summary>
      /// <param name="data">Dati</param>
      /// <param name="maxTimeInSeconds">Numero massimo di secondi di training</param>
      /// <param name="metrics">La metrica del modello migliore</param>
      /// <param name="numberOfFolds">Numero di validazioni incrociate</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il modello migliore</returns>
      public override ITransformer AutoTraining(
         IDataAccess data,
         int maxTimeInSeconds,
         out object metrics,
         int numberOfFolds = 1,
         CancellationToken cancellation = default)
      {
         var settings = new RegressionExperimentSettings
         {
            CancellationToken = cancellation,
            OptimizingMetric = BestModelSelectionMetric,
            MaxExperimentTimeInSeconds = (uint)Math.Max(0, maxTimeInSeconds)
         };
         var experiment = ML.NET.Auto().CreateRegressionExperiment(settings);
         var progress = ML.NET.RegressionProgress(Name);
         var pipes = GetPipes();
         var model = default(ITransformer);
         if (numberOfFolds > 1) {
            var experimentResult = experiment.Execute(data, (uint)Math.Max(0, numberOfFolds), LabelColumnName, null, pipes.Input, progress);
            cancellation.ThrowIfCancellationRequested();
            var best = (from r in experimentResult.BestRun.Results select (r.Model, r.ValidationMetrics)).Best();
            ML.NET.WriteLog(experimentResult.BestRun.TrainerName, Name);
            metrics = best.Metrics;
            model = best.Model;
         }
         else {
            var experimentResult = experiment.Execute(data, LabelColumnName, null, pipes.Input, progress);
            var best = experimentResult.BestRun;
            ML.NET.WriteLog(experimentResult.BestRun.TrainerName, Name);
            metrics = best.ValidationMetrics;
            model = best.Model;
         }
         if (pipes.Output != null) {
            var dataFirstRow = model.Transform(data.ToDataViewFiltered(row => row.Position == 0));
            var outputTransformer = pipes.Output.Fit(dataFirstRow);
            var result = new TransformerChain<ITransformer>(model, outputTransformer);
            return result;
         }
         else
            return model;
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
      public override ITransformer CrossValidateTraining(
         IDataAccess data,
         out object metrics,
         int numberOfFolds = 5,
         string samplingKeyColumnName = null,
         int? seed = null)
      {
         var results = ML.NET.Regression.CrossValidate(data, GetPipes().Merged, numberOfFolds, LabelColumnName ?? "Label", samplingKeyColumnName, seed);
         var best = (from r in results select (r.Model, r.Metrics)).Best();
         metrics = best.Metrics;
         return best.Model;
      }
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
         if (modelEvaluation1 is RegressionMetrics metrics1 && modelEvaluation2 is RegressionMetrics metrics2)
            best = metrics2.RSquared > metrics1.RSquared ? modelEvaluation2 : modelEvaluation1;
         return best;
      }
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="model">Modello da valutare</param>
      /// <param name="data">Dati attuali caricati</param>
      /// <returns>Il risultato della valutazione</returns>
      /// <remarks>La valutazione ottenuta verra' infine passata alla GetBestEvaluation per compaare e selezionare il modello migliore</remarks>
      protected override object GetModelEvaluation(ITransformer model, IDataAccess data) => ML.NET.Regression.Evaluate(model.Transform(data));
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="modelEvaluation">Il risultato della valutazione di un modello</param>
      /// <returns>Il risultato della valutazione in formato testo</returns>
      protected override string GetModelEvaluationInfo(object modelEvaluation)
      {
         if (modelEvaluation is not RegressionMetrics metrics)
            return null;
         var sb = new StringBuilder();
         sb.AppendLine(metrics.ToText());
         return sb.ToString();
      }
      /// <summary>
      /// Funzione di inizializzazione
      /// </summary>
      private void Init() => Trainers = new RegressionTrainersCatalog(ML);
      #endregion
   }
}
