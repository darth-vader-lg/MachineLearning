namespace MachineLearning
{
   /// <summary>
   /// Provider di contesto ML.NET
   /// </summary>
   public interface IMachineLearningContextProvider
   {
      #region Properties
      /// <summary>
      /// Contesto di machine learning
      /// </summary>
      MachineLearningContext ML { get; }
      #endregion
   }
}
