using MachineLearning.Trainers;

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
      /// Funzione di inizializzazione
      /// </summary>
      private void Init() => Trainers = new MulticlassClassificationTrainersCatalog(ML);
      #endregion
   }
}
