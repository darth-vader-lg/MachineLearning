using MachineLearning.Data;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using System;
using System.Linq;
using System.Text;
using System.Threading;
using TCatalog = MachineLearning.Trainers.CusteringCatalog;
using TMetrics = Microsoft.ML.Data.ClusteringMetrics;

namespace MachineLearning.Model
{
   /// <summary>
   /// Classe base per i classificatori binari
   /// </summary>
   [Serializable]
   public abstract class ClusteringModelBase : ModelBaseMLNet
   {
      #region Properties
      /// <summary>
      /// Nome colonna features
      /// </summary>
      public string FeaturesColumnName { get; set; } = "Features";
      /// <summary>
      /// Nome colonna label
      /// </summary>
      public string LabelColumnName { get; set; } = "Label";
      /// <summary>
      /// Catalogo di trainers
      /// </summary>
      [field: NonSerialized]
      public TCatalog Trainers { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider contesto di machine learning</param>
      public ClusteringModelBase(IContextProvider<MLContext> contextProvider = default) : base(contextProvider) =>
         Trainers = new TCatalog(this);
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
         CancellationToken cancellation = default) => throw new NotImplementedException("Autotraining is not implemented in the clustering");
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
         var results = Context.Clustering.CrossValidate(data, GetPipes().Merged, numberOfFolds, LabelColumnName ?? "Label", FeaturesColumnName ?? "Features", samplingKeyColumnName, seed);
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
            best = metrics2.AverageDistance <= metrics1.AverageDistance ? modelEvaluation2 : modelEvaluation1;
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
         Context.Clustering.Evaluate(model.Transform(data), LabelColumnName ?? "Label", "Score", FeaturesColumnName = "Features");
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
