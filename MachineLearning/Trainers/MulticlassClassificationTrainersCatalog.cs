using Microsoft.ML;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Catalogo di trainers multiclasse
   /// </summary>
   public class MulticlassClassificationTrainersCatalog : ContextProvider<MLContext>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      internal MulticlassClassificationTrainersCatalog(IContextProvider<MLContext> context) : base(context) { }
      /// <summary>
      /// Restituisce un trainer di tipo SdcaNonCalibratedMulticlass
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public SdcaNonCalibratedMulticlassTrainer SdcaNonCalibrated(Microsoft.ML.Trainers.SdcaNonCalibratedMulticlassTrainer.Options options = default) =>
         new SdcaNonCalibratedMulticlassTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo ImageClassification
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public ImageClassificationTrainer ImageClassification(Microsoft.ML.Vision.ImageClassificationTrainer.Options options = default) =>
         new ImageClassificationTrainer(this, options);
      #endregion
   }
}
