namespace MachineLearning.Model
{
   public interface IModelRetrainable
   {
      #region Properties
      /// <summary>
      /// Numero massimo di tentativi di training del modello
      /// </summary>
      int MaxTrains { get; }
      #endregion
   }
}
