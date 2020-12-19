using System;

namespace MachineLearning
{
   /// <summary>
   /// Modello per l'interpretazione del significato si testi
   /// </summary>
   [Serializable]
   public sealed partial class PredictorTextMeaning : Predictor<string>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public PredictorTextMeaning() => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Contesto di machine learning</param>
      public PredictorTextMeaning(int? seed) : base(seed) => Init();
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      public PredictorTextMeaning(MachineLearningContext ml) : base(ml) => Init();
      /// Funzione di inizializzazione
      /// </summary>
      private void Init()
      {
         DataStorage = new DataStorageTextMemory();
         ModelStorage = new ModelStorageMemory();
      }
      #endregion
   }
}
