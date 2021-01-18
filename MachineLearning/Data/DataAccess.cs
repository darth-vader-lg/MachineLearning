using Microsoft.ML;
using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;

namespace MachineLearning.Data
{
   /// <summary>
   /// Vista di dati con contesto
   /// </summary>
   public class DataAccess : IDataAccess
   {
      #region Fields
      /// <summary>
      /// Vista di dati
      /// </summary>
      private readonly IDataView _data;
      #endregion
      #region Properties
      /// <summary>
      /// Possibilita' di shuffle
      /// </summary>
      public bool CanShuffle => _data.CanShuffle;
      /// <summary>
      /// Contesto
      /// </summary>
      public MachineLearningContext ML { get; }
      /// <summary>
      /// Lo schema dei dati
      /// </summary>
      public DataViewSchema Schema => _data.Schema;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">Vista di dati</param>
      public DataAccess(IMachineLearningContextProvider context, IDataView data)
      {
         MachineLearningContext.CheckMLNET(context, nameof(context));
         context.ML.NET.CheckValue(data, nameof(data));
         ML = context.ML;
         _data = data;
      }
      /// <summary>
      /// Restituisce il numero di righe
      /// </summary>
      /// <returns>Il numero di righe</returns>
      public long? GetRowCount() =>
         _data.GetRowCount();
      /// <summary>
      /// Restituisce un cursore
      /// </summary>
      /// <param name="columnsNeeded">Le colonne richieste</param>
      /// <param name="rand">Eventuale generatore di numeri casuali</param>
      /// <returns>Il cursore</returns>
      public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null) =>
         _data.GetRowCursor(columnsNeeded, rand);
      /// <summary>
      /// Restituisce un set di cursori
      /// </summary>
      /// <param name="columnsNeeded">Le colonne richieste</param>
      /// <param name="n">Il numero desiderato di cursori</param>
      /// <param name="rand">Eventuale generatore di numeri casuali</param>
      /// <returns>I cursori</returns>
      public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null) =>
         _data.GetRowCursorSet(columnsNeeded, n, rand);
      #endregion
   }
}
