using MachineLearning.Trainers;

namespace MachineLearning
{
   /// <summary>
   /// Classe base per i previsori a regressione
   /// </summary>
   public abstract class PredictorRegression : Predictor
   {
      #region Properties
      /// <summary>
      /// Catalogo di trainers
      /// </summary>
      public RegressionTrainersCatalog Trainers { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public PredictorRegression() : base() => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Seme operazioni random</param>
      public PredictorRegression(int? seed) : base(seed) => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public PredictorRegression(MachineLearningContext ml) : base(ml) => Init();
      /// <summary>
      /// Funzione di inizializzazione
      /// </summary>
      private void Init() => Trainers = new RegressionTrainersCatalog(ML);
      #endregion
   }
}
