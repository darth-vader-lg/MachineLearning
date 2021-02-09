﻿using MachineLearning.Data;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using System;
using System.Linq;
using System.Text;
using System.Threading;
using TExperimentSettings = Microsoft.ML.AutoML.RankingExperimentSettings;
using TMetrics = Microsoft.ML.Data.RankingMetrics;
using TTrainers = MachineLearning.Trainers.RankingTrainers;

namespace MachineLearning.Model
{
   /// <summary>
   /// Classe base per i classificatori binari
   /// </summary>
   [Serializable]
   public abstract class RankingModelBase : ModelBaseMLNet
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
      /// Nome colonna di identificativo di raggruppamento
      /// </summary>
      public string GroupIdColumnName { get; set; } = "GroupId";
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
      public RankingModelBase(IContextProvider<MLContext> contextProvider = default) : base(contextProvider) =>
         Trainers = new TTrainers(this);
      /// <summary>
      /// Effettua il training con la ricerca automatica del miglior trainer
      /// </summary>
      /// <param name="data">Dati</param>
      /// <param name="maxTimeInSeconds">Numero massimo di secondi di training</param>
      /// <param name="metrics">La metrica del modello migliore</param>
      /// <param name="numberOfFolds">Numero di validazioni incrociate</param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il modello migliore</returns>
      public override sealed ITransformer AutoTraining(
         IDataAccess data,
         int maxTimeInSeconds,
         out object metrics,
         int numberOfFolds = 1,
         CancellationToken cancellation = default)
      {
         autoTrainingTask ??= new AutoTrainingTask<TMetrics, TExperimentSettings>(this);
         var result = autoTrainingTask.WaitResult(
            () => Context.Auto().CreateRankingExperiment(new TExperimentSettings
            {
               CancellationToken = cancellation,
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
      public override sealed ITransformer CrossValidateTraining(
         IDataAccess data,
         out object metrics,
         int numberOfFolds = 5,
         string samplingKeyColumnName = null,
         int? seed = null)
      {
         var results = Context.Ranking.CrossValidate(data, GetPipes().Merged, numberOfFolds, LabelColumnName ?? "Label", GroupIdColumnName, seed);
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
         if (modelEvaluation1 is TMetrics metrics1 && modelEvaluation2 is TMetrics metrics2)
            best = RankingClassificationExtensions.DcgScore(metrics2.NormalizedDiscountedCumulativeGains) >= RankingClassificationExtensions.DcgScore(metrics1.NormalizedDiscountedCumulativeGains) ? modelEvaluation2 : modelEvaluation1;
         return best;
      }
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="model">Modello da valutare</param>
      /// <param name="data">Dati attuali caricati</param>
      /// <returns>Il risultato della valutazione</returns>
      /// <remarks>La valutazione ottenuta verra' infine passata alla GetBestEvaluation per compaare e selezionare il modello migliore</remarks>
      protected override object GetModelEvaluation(ITransformer model, IDataAccess data) =>
         Context.Ranking.Evaluate(model.Transform(data), LabelColumnName ?? "Label", GroupIdColumnName);
      /// <summary>
      /// Funzione di restituzione della valutazione del modello (metrica, accuratezza, ecc...)
      /// </summary>
      /// <param name="modelEvaluation">Il risultato della valutazione di un modello</param>
      /// <returns>Il risultato della valutazione in formato testo</returns>
      protected override string GetModelEvaluationInfo(object modelEvaluation)
      {
         if (modelEvaluation is TMetrics metrics) {
            var sb = new StringBuilder();
            sb.AppendLine(metrics.ToText());
            return sb.ToString();
         }
         return null;
      }
      #endregion
   }
}
