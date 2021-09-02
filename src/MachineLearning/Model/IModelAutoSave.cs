namespace MachineLearning.Model
{
   /// <summary>
   /// Interface for models with autosave capability
   /// </summary>
   public interface IModelAutoSave
   {
      #region Properties
      /// <summary>
      /// Enable automatic save of updated model
      /// </summary>
      bool ModelAutoSave { get; }
      #endregion
   }
}
